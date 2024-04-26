"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loaders.loader_parquet import LoaderDS
from loaders.loader_hf import LoaderHF
import numpy as np
from models.encoder import Encoder
from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.heads import SequenceHead, ClassifierHead
from models.decoder import DenovoDecoder, DenovoDiffusionDecoder
import os
from tqdm import tqdm
from collections import deque
from time import time
import utils as U
from copy import deepcopy
nn = th.nn
F = nn.functional
choice = np.random.choice
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class DownstreamObj:
    def __init__(self, config, task='denovo_ar', base_model=None, svdir='./downstream/'):
        
        # Config is entire downstream yaml
        self.config = config
        self.task = task
        
        # Create directory for saving results; only use if run from PretrainModel.py
        self.log = config['log'] or config['save_weights']
        self.header = config['header']
        if svdir[-1] != '/': svdir += '/'
        if self.log and not os.path.exists(svdir):
            os.makedirs(svdir)
        if self.config['save_weights']:
            if not os.path.exists(svdir+'weights'):
                os.mkdir(svdir+'weights')
        self.svdir = svdir
       
        if config['lr_warmup']:
            self.lr_warmup_increment = (
                (config['lr_warmup_end']-config['lr_warmup_start']) / 
                config['lr_warmup_steps']
            )
            self.starting_lr = config['lr_warmup_start']
        else:
            self.starting_lr = config['lr']

        # Base model
        # - must do base model beforehand dataloader in order to transfer over 
        #   its configuration settings to self.config
        self.configure_encoder(base_model) # self.config updated
        #if not config['train_encoder']: self.encoder.trainable = False

        self.running_loss = []
        self.global_step = 0
    
    def configure_encoder(self, imported_encoder=None):
        
        self.imported = True if imported_encoder is not None else False
        
        # If encoder model passed in as argument OR no saved weights
        # Assert that the dsconfig top_pks has the encoder's value
        if self.imported:
            assert self.config['loader']['top_pks'] == imported_encoder.sl
            #self.config['loader']['top_pks'] == imported_encoder.sl
            self.encoder = imported_encoder

        else:
            # If no saved pretraining model
            # Get configuration settings from current pretrain yaml files
            if self.config['pretrain_path'] is None:
                yaml_config_path = './yaml/config.yaml'
                yaml_model_path = './yaml/models.yaml'
        
            # If encoder is loaded from saved pretraining path
            # Get configuration settings from saved experiment
            else:
                assert os.path.exists(self.config['pretrain_path'])
                yaml_config_path = self.config['pretrain_path']+'/yaml/config.yaml'
                yaml_model_path = self.config['pretrain_path']+'/yaml/models.yaml'
                if self.config['dswts']:
                    assert os.path.exists(self.config['pretrain_path']+'/dswts')
                    weights_path = self.config['pretrain_path']+'/dswts/encoder.wts'
                else:
                    weights_path = self.config['pretrain_path']+'/weights/model_enc.wts'
        
            # Open yaml files
            with open(yaml_config_path) as stream:
                ptconf = yaml.safe_load(stream)
            with open(yaml_model_path) as stream:
                ptmodconf = yaml.safe_load(stream)
            # Transfer over settings to self.config
            self.config['loader']['top_pks'] = ptconf['max_peaks']
            self.config['encoder_dict'] = ptmodconf['encoder_dict']
            
            # ENCODER TYPE
            if ptmodconf['encoder_name'] == 'depthcharge':
                self.encoder = dc_encoder(sequence_length=ptconf['max_peaks'])
            else:
                self.encoder = Encoder(**self.config['encoder_dict'], device=device)

            if self.config['pretrain_path'] is not None:
                self.encoder.load_state_dict(th.load(weights_path, map_location=device))
        
        self.encoder.to(device)
        self.opt_encoder = th.optim.Adam(
            self.encoder.parameters(), self.starting_lr
        )

        print(f"Total Encoder parameters: {self.encoder.total_params():,}") 

    def save_head(self, fp='./head.wts'):
        th.save(self.head.state_dict(), fp)
    
    def save_encoder(self, fp='./encoder.wts'):
        th.save(self.encoder.state_dict(), fp)

    def save_all_weights(self, der='./'):
        self.save_head(der=der+'head.wts')
        self.save_encoder(der=der+'encoder.wts')

    def split_labels_str(self, incl_str):
        return [label for label in self.dl.labels if incl_str in label]

    def encinp(self, 
               batch, 
               mask_length=True, 
               return_mask=False, 
               ):

        mzab = th.cat([batch['mz'][...,None], batch['ab'][...,None]], -1)
        model_inp = {
            'x': mzab.to(device),
            'charge': (
                batch['charge']
                if self.config['encoder_dict']['use_charge'] else 
                None
            ),
            'mass': (
                batch['mass']
                if self.config['encoder_dict']['use_mass'] else
                None
            ),
            'length': batch['length'] if mask_length else None,
            'return_mask': return_mask,
        }

        return model_inp
    
    def call(self, enc_inp_dict, training=False):
        if training: 
            self.encoder.train()
            self.head.train()
        else: 
            self.encoder.eval()
            self.head.eval()
        
        embedding = self.encoder(**enc_inp_dict)['emb']
        out = self.head(embedding)

        return out
    
    def LossFunction(self, target, prediction):
        targ_one_hot = F.one_hot(target, self.predcats).type(th.float32)
        all_loss = F.cross_entropy(prediction, targ_one_hot)
        
        return all_loss
    
    def train_step(self, batch, trenc=True):
        U.Dict2dev(batch, device)
        enc_input, target = self.inptarg(batch)
        
        if trenc:
            self.encoder.train()
            self.encoder.zero_grad()
            embedding = self.encoder(**enc_input)['emb']
        else:
            self.encoder.eval()
            with th.no_grad():
                embedding = self.encoder(**enc_input)['emb']
        
        self.head.train()
        self.head.decoder.zero_grad()
        head_out = self.head(embedding)
        all_loss = self.LossFunction(target, head_out)
        loss = all_loss.mean()
        
        loss.backward()
        
        self.opt_head.step()
        if trenc:
            self.opt_encoder.step()

        return loss

    def train_epoch(self, svfreq=10000):
        
        bs = self.config['batch_size']
        running_loss = deque(maxlen=50)
        running_time = [deque(maxlen=50) for _ in range(5)];running_time[-1].append(0)
        
        epoch_start = time()
        step_end=epoch_start;split1=epoch_start;split2=epoch_start;split3=epoch_start
        for step, batch in enumerate(self.data.dataloader['train']):
            step_start = time()
            running_time[0].append(step_start - step_end)

            # Are we training the encoder? Two conditions must be met.
            train_encoder = (
                True 
                if (
                    self.config['train_encoder'] and 
                    (self.global_step >= self.config['encoder_start'])
                ) else 
                False
            ) # boolean argument into train_step
            loss = self.train_step(batch, train_encoder)
            self.global_step += 1
            split1 = time()
            
            running_loss.append(loss.detach().cpu())
            split2 = time()
            
            rlm = np.mean(running_loss)
            rtm = np.mean(running_time[-1]) #[np.mean(m) if len(m)>0 else 0 for m in running_time]
            split3 = time()
            print("\rTraining step %d  Running Loss: %.6f (%.3f s)"%(step+1, rlm, rtm), end='')
            
            self.running_loss.append(rlm)
            if self.log and (self.global_step % svfreq == 0):
                self.savetxt(self.running_loss)
                self.running_loss = []

            running_time[4].append(time() - step_end)
            step_end = time()
            
            #running_time[1].append(split1 - step_start)
            #running_time[2].append(split2 - split1)
            #running_time[3].append(split3 - split2)
        
        if self.log and (len(self.running_loss) > 0):
            self.savetxt(self.running_loss)
            self.running_loss = []
        
        print("\rFinal running loss: %.6f, Final time elapsed: %.0f s"%(rlm, time()-epoch_start))
        
    def savetxt(self, train_loss=None, eval_stats=None):
        if eval_stats is not None:
            np.savetxt(self.svdir+"eval_stats.txt", np.array(eval_stats), fmt='%.6f', header=self.header)
        if train_loss is not None:
            if os.path.exists(self.svdir+"train_loss.txt"):
                train_loss = np.append(np.loadtxt(self.svdir+"train_loss.txt"), train_loss)
            np.savetxt(self.svdir+"train_loss.txt", train_loss, fmt="%.6f", header=self.header)

class BaseDenovo(DownstreamObj):
    def __init__(self, 
                 config, 
                 task='denovo', 
                 base_model=None, 
                 ar=False, 
                 svdir='./downstream/'
                 ):
        super().__init__(config=config, task=task, base_model=base_model, svdir=svdir)
        self.ar = ar
 
        # Dataloader
        self.data = LoaderHF(**self.config['loader'])
        self.predcats = np.max(list(self.data.amod_dic.values())) + 1

        self.eval_stats = []

    def evaluation(self, dset='val'):
        
        func = self.head.predict_sequence if self.ar else self.call
        
        # counters
        #totsz = self.dl.dfs[dset].shape[0]
        #steps = totsz // self.config['batch_size']
        #steps += 0 if (totsz % self.config['batch_size'])==0 else 1
        
        # losses
        out = {'ce': 0, 'old_recall': 0, 'recall': 0, 'precision': 0, 'auprc': 0}
        old_recall_sum = 0
        tots = {
            'recall': 0,
            'precision': 0,
        }
        
        self.encoder.eval()
        self.head.eval()
        for i, batch in enumerate(self.data.dataloader[dset]):
            print("\rEvaluation step %d"%(i+1), end='')
            batch = U.Dict2dev(batch, device)
            # Fork in the code for the 2 types of denovo models I created
            with th.no_grad():
                enc_input, seqint, target, loss_mask = self.inptarg(
                    batch, #full_seqint=True,
                )
                embedding = self.encoder(**enc_input)
                prediction, probs = self.head.predict_sequence(embedding, batch)
                pred = probs.transpose(-1,-2)
            
            out['ce'] += (
                F.cross_entropy(pred, target, reduction='none')[loss_mask].sum()
                if self.ar else 
                self.LossFunction(target, pred, loss_mask).sum()
            )
                        
            vecs, auprc = U.RocCurve(target, prediction, probs, null_value=self.head.outdict['<EOS>'], typ='aa')
            out['old_recall'] += U.roc_apply_threshold(**vecs, threshold=0)['recall']*vecs['precision'].shape[0]
            old_recall_sum += vecs['precision'].shape[0]
            out['auprc'] += auprc
            roc_stats = U.roc_apply_threshold(**vecs, threshold=0.9)
            for metric in roc_stats.keys():
                tots[metric] += roc_stats[metric]
        
        steps = i+1
        totsz = self.config['loader']['batch_size']*steps
        out['ce'] = float((out['ce'] / (totsz * self.config['sl'])).cpu().detach().numpy())
        out['old_recall'] = out['old_recall'] / old_recall_sum
        out['auprc'] = out['auprc'] / steps
        for metric in roc_stats.keys():
            out[metric] = tots[metric] /  steps

        return out

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        highscore = 0
        for i in range(self.config['epochs']):
            self.data.dataset['train'].set_epoch(i)
            self.train_epoch()
            
            out = self.evaluation(dset=eval_dset)
            
            line = "ValEpoch %d: Cross-entropy=%.4f, Recall(0)=%.4f, Recall(90)=%.4f, Precision(90)=%.4f, AUPRC=%.4f"%(
                (i,) + tuple(out.values())
            )
            if out['recall']>highscore:
                highline = line
                highscore = out['recall']
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            print("\r"+line)

            if self.config['save_weights']:
                self.save_head(self.svdir+'weights/head.wts')
                if self.config['train_encoder']:
                    self.save_encoder(self.svdir+'weights/encoder.wts')
            
            self.eval_stats.append(list(out.values()))
            
            # Save data
            if self.log:
                self.savetxt(train_loss=None, eval_stats=np.array(self.eval_stats))
        
        return lines, highline

class DenovoArDSObj(BaseDenovo):
    def __init__(self, config, base_model=None, svdir='./dswts/'):
        task = 'denovo_ar'
        super().__init__(
            config=config, task=task, base_model=base_model, ar=True, 
            svdir=svdir
        )

        # Head model
        head_dict = self.config[task]['head_dict']
        head_dict['kv_indim'] = self.encoder.run_units
        self.config['sl'] = self.config['loader']['pep_length'][1]
        self.head = DenovoDecoder(
            token_dict=self.data.amod_dic, dec_config=head_dict, 
            encoder=self.encoder # encoder is set by inherited class
        )
        print(f"Total Decoder parameters: {self.head.decoder.total_params():,}")
        if config['pretrain_path'] is not None and os.path.exists(self.svdir + '/head.wts'):
            self.head.load_weights(self.svdir + '/head.wts', device)
        self.head.decoder.to(device)
        
        self.opt_head = th.optim.Adam(self.head.parameters(), self.starting_lr)

    def append_null_token(self, intseq):
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.head.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def replace_with_eos_token(self, intseq, lengths):
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.head.EOS

        return intseq
    
    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])
        
        enc_input = self.encinp(batch, return_mask=True)

        dec_input = self.head.prepend_startok(dec_input)

        target = self.append_null_token(target)
        target = self.replace_with_eos_token(target, batch['peplen'])

        loss_mask = self.head.decoder.sequence_mask(batch['peplen'], target.shape[1])
        loss_mask = loss_mask == 0

        return enc_input, dec_input, target, loss_mask

    def LossFunction(self, target, prediction, loss_mask):
        targ_one_hot = F.one_hot(target, self.predcats).type(th.float32)
        targ_one_hot = targ_one_hot.transpose(-1,-2)
        prediction = prediction.transpose(-1,-2)
        all_loss = F.cross_entropy(prediction, targ_one_hot, reduction='none')
        masked_loss = all_loss[loss_mask]
        loss = masked_loss.sum() / loss_mask.sum()

        return loss

    def train_step(self, batch, trenc=True):
        batch = U.Dict2dev(batch, device)
        #enc_input, seqint, target = self.inptarg(batch)
        enc_input, dec_input, target, loss_mask = self.inptarg(batch)

        self.encoder.to(device)
        if trenc:
            self.encoder.train()
            self.encoder.zero_grad()
            embedding = self.encoder(**enc_input)
        else:
            self.encoder.eval()
            with th.no_grad():
                embedding = self.encoder(**enc_input)
        
        self.head.train()
        self.head.decoder.zero_grad()
        head_out = self.head(dec_input, embedding, batch, training=True)
        all_loss = self.LossFunction(target, head_out, loss_mask)
        loss = all_loss.mean()
        
        loss.backward()
        
        if self.config['lr_warmup']:
            if self.global_step < self.config['lr_warmup_steps']:
                self.opt_head.param_groups[-1]['lr'] += self.lr_warmup_increment
                self.opt_encoder.param_groups[-1]['lr'] += self.lr_warmup_increment

        self.opt_head.step()
        if trenc:
            self.opt_encoder.step()
        
        return loss

from models.diffusion.model_utils import create_model_and_diffusion

class DenovoDiffusionObj(BaseDenovo):
    def __init__(self, config, base_model=None, svdir='./dswts/'):
        task = 'denovo_diff'
        super().__init__(
            config=config, task=task, base_model=base_model, ar=False, 
            svdir=svdir
        )

        # Head model
        head_dict = self.config[task]['head_dict']
        head_dict['kv_indim'] = self.encoder.run_units
        self.config['sl'] = self.config['loader']['pep_length'][1]
        self.head = DenovoDiffusionDecoder(
            token_dict=self.data.amod_dic, 
            dec_config=head_dict, 
            self_condition=config['denovo_diff']['self_condition'],
            **config['denovo_diff']['head_dict'],
        )
        print(f"Total Decoder parameters: {self.head.decoder.total_params():,}")
        if config['pretrain_path'] is not None and os.path.exists(self.svdir + '/head.wts'):
            self.head.load_weights(self.svdir + '/head.wts', device)
        self.head.to(device)

        # Diffusion object
        with open("./yaml/diffusion.yaml") as stream:
            diff_config = yaml.safe_load(stream)
        diff_config['pad_tok_id'] = self.head.NT
        diff_config['resume_checkpoint'] = False
        diff_config['sequence_len'] = self.config['loader']['pep_length'][1] + 1 # b/c of eos token
        _, self.diff_obj = create_model_and_diffusion(**diff_config)

    def append_null_token(self, intseq):
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.head.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def replace_with_eos_token(self, intseq, lengths):
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.head.EOS

        return intseq
    
    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])

        # Schedule sampler
        timesteps = th.empty(bs).uniform_(0, self.diff_obj.num_timesteps).type(th.int32).to(target.device)
        
        enc_input = self.encinp(batch, return_mask=True)

        target = self.append_null_token(target)
        target = self.replace_with_eos_token(target, batch['peplen'])

        loss_mask = self.head.decoder.sequence_mask(batch['peplen'], target.shape[1])
        loss_mask = loss_mask == 0

        return enc_input, timesteps, target, loss_mask

    def train_step(self, batch, trenc=True):
        batch = U.Dict2dev(batch, device)
        enc_input, timesteps, target, loss_mask = self.inptarg(batch)

        self.encoder.to(device)
        if trenc:
            self.encoder.train()
            self.encoder.zero_grad()
            embedding = self.encoder(**enc_input)
        else:
            self.encoder.eval()
            with th.no_grad():
                embedding = self.encoder(**enc_input)
        
        self.head.train()
        self.head.decoder.zero_grad()
        model_kwargs = {
            'input_ids': None,
            'decoder_input_ids': target,
            'loss_mask': loss_mask,
            'kv_feats': embedding['emb'],
        }
        losses = self.diff_obj.training_losses(
            self.head, 
            self.global_step,
            timesteps, 
            model_kwargs=model_kwargs, 
            noise=None
        )
        
        loss.backward()
        
        if self.config['lr_warmup']:
            if self.global_step < self.config['lr_warmup_steps']:
                self.opt_head.param_groups[-1]['lr'] += self.lr_warmup_increment
                self.opt_encoder.param_groups[-1]['lr'] += self.lr_warmup_increment

        self.opt_head.step()
        if trenc:
            self.opt_encoder.step()
        
        return loss

if __name__ == '__main__':

    # Read downstream yaml
    with open("./yaml/config.yaml") as stream:
        config = yaml.safe_load(stream)
    with open("./yaml/downstream.yaml") as stream:
        dsconfig = yaml.safe_load(stream)
        dsconfig['log'] = config['log']
        dsconfig['header'] = config['header']
    
    # Shorthand
    bs = dsconfig['batch_size']
    msg = config['log']
    swt = config['svwts']
    
    # Create experiment directory in save/downstream_only/
    if (msg or swt):
        timestamp = U.timestamp()
        svdir = 'save/downstream_only/' + timestamp
        U.create_experiment(svdir, svwts=config['svwts'])
        with open(svdir + '/experiment_header', 'w') as f:
            f.write("Experiment header: " + config['header'])
        print("Experiment is writing to directory %s"%svdir)
    else:
        svdir = './'

    # Downstream object
    print("Denovo sequencing")
    D = DenovoDiffusionObj(dsconfig, svdir=svdir)
    #out = D.evaluation(dset='val')
    print(D.TrainEval()[-1])
