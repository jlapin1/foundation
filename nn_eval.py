"""
Find, for every spectrum in a test set, the top-n other spectra whose encoder
embeddings are most similar to it.

Two batching stages, to keep memory bounded regardless of test-set size:
  1. compute_embeddings: runs the encoder over the test set one dataloader
     batch at a time and mean/max-pools each (seq_len, features) output down
     to a single per-spectrum vector.
  2. nearest_neighbors_to_parquet: compares embeddings in chunks of
     `query_batch_size` rows against the full reference set (instead of
     materializing the full N x N similarity matrix) and streams each
     chunk's top-n results straight to parquet.

evaluate_similarity ties the two together and is the function meant to be
called from PretrainModel.py's training loop.
"""
import numpy as np
import torch as th
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

F = th.nn.functional


def _masked_pool(emb, length, mode='mean'):
    bs, sl, _ = emb.shape
    idx = th.arange(sl, device=emb.device)[None, :]
    valid = idx < length[:, None].to(emb.device)

    if mode == 'mean':
        mask = valid.unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
    elif mode == 'max':
        masked = emb.masked_fill(~valid.unsqueeze(-1), float('-inf'))
        return masked.max(dim=1).values.nan_to_num(neginf=0.0)
    raise ValueError(f"Unknown pooling mode: {mode!r}")


def compute_embeddings(encoder, dataobj, device, pooling='mean'):
    """Run `encoder` over every batch in `dataloader`, pooling each
    (batch, seq_len, features) output to (batch, features).

    Returns (embeddings: FloatTensor[N, features] on CPU, ids: ndarray[N]).
    """
    was_training = encoder.training
    encoder.eval()
    embeddings, ids, ilocs = [], [], []
    pbar = tqdm(dataobj.dataloader['val'], leave=False)
    pbar.set_description("Validation")
    with th.no_grad():
        for batch in pbar:
            mzab = th.cat([batch['mz'][..., None], batch['ab'][..., None]], dim=-1).to(device)
            inp = {
                'x': mzab,
                'charge': batch['charge'].to(device),
                'mass': batch['mass'].to(device),
                'length': batch['length'].to(device),
            }
            emb = encoder(**inp)['emb']
            pooled = _masked_pool(emb, inp['length'], mode=pooling)
            embeddings.append(pooled.cpu())
            ids.append(np.asarray(batch['name']))
            ilocs.append(batch['iloc'])
    if was_training:
        encoder.train()

    return th.cat(embeddings, dim=0), np.concatenate(ids), np.concatenate(ilocs)


def nearest_neighbors_to_parquet(
    embeddings, ids, ilocs, output_path,
    top_n=10, metric='cosine', query_batch_size=2048,
    exclude_self=True, device=None,
):
    """For every row in `embeddings`, find its top-n closest other rows and
    stream the results to a parquet file with columns:
    query_id, rank (1 = closest), neighbor_id, score.

    Only ever materializes a (query_batch_size, N) score matrix at a time,
    never the full (N, N) matrix, so this scales to tens of thousands (or
    more) of test-set examples.
    """
    assert metric in ('cosine', 'euclidean')
    device = device or th.device('cuda' if th.cuda.is_available() else 'cpu')
    N = embeddings.shape[0]
    ids = np.asarray(ids)

    ref = embeddings.to(device)
    if metric == 'cosine':
        ref = F.normalize(ref, dim=-1)
    largest = metric == 'cosine'

    eff_top_n = min(top_n, N - 1) if exclude_self else min(top_n, N)
    k = eff_top_n + (1 if exclude_self else 0)

    schema = pa.schema([
        ('query_id', pa.string()),
        ('rank', pa.int32()),
        ('neighbor_id', pa.string()),
        ('neighbor_iloc', pa.int32()),
        ('score', pa.float32()),
    ]).with_metadata({'metric': metric})

    writer = pq.ParquetWriter(output_path, schema)
    try:
        for start in range(0, N, query_batch_size):
            end = min(start + query_batch_size, N)
            q = ref[start:end]

            if metric == 'cosine':
                scores = q @ ref.T
            else:
                scores = th.cdist(q, ref)

            if exclude_self:
                local_idx = th.arange(end - start, device=device)
                global_idx = local_idx + start
                scores[local_idx, global_idx] = float('-inf') if largest else float('inf')

            top_scores, top_idx = scores.topk(k, dim=1, largest=largest)
            if exclude_self:
                top_scores, top_idx = top_scores[:, :eff_top_n], top_idx[:, :eff_top_n]

            top_scores = top_scores.cpu().numpy().reshape(-1).astype(np.float32)
            top_idx = top_idx.cpu().numpy().reshape(-1)
            n_rows = end - start

            table = pa.table({
                'query_id': np.repeat(ids[start:end], eff_top_n).astype(str),
                'rank': np.tile(np.arange(1, eff_top_n + 1, dtype=np.int32), n_rows),
                'neighbor_id': ids[top_idx].astype(str),
                'neighbor_iloc': ilocs[top_idx],
                'score': top_scores,
            }, schema=schema)
            writer.write_table(table)
    finally:
        writer.close()


def evaluate_similarity(
    encoder, dataobj, output_path,
    top_n=10, pooling='mean', metric='cosine',
    encode_device=None, search_device=None,
    query_batch_size=2048, exclude_self=True,
):
    """Encode the whole test set with `encoder`, then write each example's
    top-n nearest neighbors to `output_path` (parquet).
    """
    encode_device = encode_device or next(encoder.parameters()).device
    embeddings, ids, ilocs = compute_embeddings(encoder, dataobj, encode_device, pooling=pooling)

    nearest_neighbors_to_parquet(
        embeddings, ids, ilocs, output_path,
        top_n=top_n, metric=metric, query_batch_size=query_batch_size,
        exclude_self=exclude_self, device=search_device or encode_device,
    )
