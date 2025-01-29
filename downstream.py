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
from models.diff_decoder import DenovoDiffusionDecoder
from models.decoder import DenovoDecoder
import os
from tqdm import tqdm
from collections import deque
from time import time
import utils as U
from copy import deepcopy
import wandb
from glob import glob
import metrics as met
import pandas as pd
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
        self.log = config['save_weights']
        self.header = "HEADER"#config['header']
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
                    assert os.path.exists(self.config['pretrain_path']+'/weights')
                    weights_path = self.config['pretrain_path']+'/weights/encoder.wts'
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
            
            # DOWNSTREAM ONLY
            if self.config['dswts'] is not None:
                regex = '*encoder*last*wts*' if self.config['load_last'] else "*encoder*wts*"
                print(f"<DSCOMMENT> Searching for decoder weights with regular expression {regex}")
                weights_path = glob(os.path.join(self.svdir, "weights", regex))
                if len(weights_path) > 1:
                    try:
                        weights_path = [m for m in weights_path if 'high' in m][0]
                        qualifier = '"high"'
                    except:
                        weights_path = [m for m in weights_path if 'last' in m][0]
                        qualifier = '"last"'
                else:
                    weights_path = weights_path[0]
                    qualifier = 'last' if self.config['load_last'] else 'only'
                print(f"<DSCOMMENT> Loading {qualifier} previous encoder weights")
                self.encoder.load_state_dict(th.load(weights_path, map_location=device))
        
        self.encoder.to(device)
        self.opt_encoder = th.optim.Adam(
            self.encoder.parameters(), self.starting_lr
        )

        print(f"<DSCOMMENT> Total Encoder parameters: {self.encoder.total_params():,}") 

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
        self.head.zero_grad()
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
        running_loss = {key: deque(maxlen=20) for key in self.training_loss_keys}
        #running_time = [deque(maxlen=50) for _ in range(5)];running_time[-1].append(0)
        
        # Progress bar
        train_steps = int(self.data.train_size // bs)
        pbar = tqdm(self.data.dataloader['train'], total=train_steps)

        epoch_start = time()
        step_end=epoch_start;split1=epoch_start;split2=epoch_start;split3=epoch_start
        for step, batch in enumerate(pbar):
            
            step_start = time()
            #running_time[0].append(step_start - step_end)
            
            if self.config['log_wandb']: wandb.log({"Learning rate": self.opt_encoder.param_groups[-1]['lr']})

            # Are we training the encoder? Two conditions must be met.
            train_encoder = (
                True if (
                    self.config['train_encoder'] and 
                    (self.global_step >= self.config['encoder_start'])
                ) else False
            ) # boolean argument into train_step
            
            losses = self.train_step(batch, train_encoder)
            self.global_step += 1
            split1 = time()
            
            for key in running_loss.keys(): running_loss[key].append(losses[key].detach().cpu())
            split2 = time()
            
            rlm = {key: np.mean(running_loss[key]) for key in running_loss.keys()}
            #rtm = np.mean(running_time[-1]) #[np.mean(m) if len(m)>0 else 0 for m in running_time]
            split3 = time()
            if self.config['log_wandb']:
                loss_printout = 'Loss: %7f'%rlm['loss']
            else:
                loss_printout = ", ".join(len(rlm)*['%s: %7f'])%tuple([m for n in rlm.items() for m in n])
            pbar.set_description(f"Running Loss: {loss_printout}")

            if self.config['log_wandb']:
                global_grad_norm_encoder = U.global_grad_norm(self.encoder)
                global_grad_norm_decoder = U.global_grad_norm(self.head)
                self.log_wandb(losses, rlm, global_grad_norm_encoder, global_grad_norm_decoder)
                

            self.running_loss.append(rlm['loss'])
            if self.log and (self.global_step % svfreq == 0):
                self.savetxt(self.running_loss)
                self.running_loss = []

            #running_time[4].append(time() - step_end)
            step_end = time()
            
            #running_time[1].append(split1 - step_start)
            #running_time[2].append(split2 - split1)
            #running_time[3].append(split3 - split2)

            #if step == 4000:
            #    break
        
        if self.log and (len(self.running_loss) > 0):
            self.savetxt(self.running_loss)
            self.running_loss = []
        
        print("\rFinal running loss: %s, Final time elapsed: %.0f s"%(loss_printout, time()-epoch_start))
        
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
        if 'val_steps' in self.config['loader'].keys(): # backwards compatibility
            val_steps = self.config['loader']['val_steps']
            self.val_steps = 1 if val_steps == None else val_steps # backwards compatiblity
        else:
            self.val_steps = 100
        self.data = LoaderHF(**self.config['loader'])
        self.predcats = np.max(list(self.data.amod_dic.values())) + 1

        self.eval_stats = []

    def replace_with_eos_token(self, intseq, lengths):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.head.EOS

        return intseq

    def append_null_token(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.head.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def fill_null_after_first_eos_token(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        length = ((intseq == self.head.EOS)|(intseq == self.head.NT)).int().argmax(1)
        mask = length > 0
        index_array = th.arange(sl)[None].repeat([sum(mask), 1]).to(intseq.device)
        boolean_array = index_array > length[mask, None]
        intseq[mask][boolean_array] = self.head.NT

        return intseq

    def to_list_of_strings(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        return [
            [
                self.head.rev_outdict[int(n)] 
                for n in m if n not in [self.head.NT, self.head.EOS]
            ] 
            for m in intseq
        ]

    def evaluation(self, dset='val', max_batches=1e10, save_df=False):
        
        func = self.head.predict_sequence if self.ar else self.call
        
        # Dataframe
        if save_df:
            dataframe = {
                'targ_intseq': [],
                'charge': [],
                'mass': [],
                'peptide_length': [],
                'pred_intseq': [],
                'probs': [],
                'targ_aaseq': [],
                'pred_aaseq': [],
                'correct_aa': [],
                'correct_peptide': [],
            }

        # losses
        out = {'ce': 0}
        tots = {'sum':{}, 'total': {}}

        # Progress bar
        val_steps = min(
            self.data.val_size // self.data.dataloader[dset].batch_size,
            max_batches,
        )
        pbar = tqdm(self.data.dataloader[dset], total=val_steps, leave=False)
        
        self.encoder.eval()
        self.head.eval()
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Evaluation")
            if i == max_batches:
                break

            #print("\rEvaluation step %d"%(i+1), end='')
            batch = U.Dict2dev(batch, device)
            with th.no_grad():
                enc_input, seqint, target, loss_mask = self.inptarg(batch)
                embedding = self.encoder(**enc_input)
                prediction, probs = self.head.predict_sequence(embedding, batch)
            
            # Do some resizing/reshaping
            prediction = prediction[..., :target.shape[1]] # loaded shapes can change based on batch
            probs = probs[:, :target.shape[1]]
            predicted_probs = probs.softmax(-1).gather(-1, prediction[...,None].type(th.int64)).squeeze()
            pred = probs.transpose(-1,-2)
            
            # Cross entropy
            out['ce'] += (
                F.cross_entropy(pred, target, reduction='none')[loss_mask].sum()
            )
            
            # Deepnovo metrics
            prediction = self.fill_null_after_first_eos_token(prediction)
            pred_strings = self.to_list_of_strings(prediction)
            targ_strings = self.to_list_of_strings(target)
            aa_matches_batch, n_aa1, n_aa2 = met.aa_match_batch(pred_strings, targ_strings, self.data.massdic)
            dn_metrics = {
                'sum': {
                    'aa_recall': sum([sum(m[0]) for m in aa_matches_batch]),
                    'aa_precision': sum([sum(m[0]) for m in aa_matches_batch]),
                    'peptide': sum([m[-1] for m in aa_matches_batch]),
                },
                'total': {
                    'aa_recall': n_aa2,
                    'aa_precision': n_aa1,
                    'peptide': len(pred_strings),
                },
            }

            if save_df:
                dataframe['charge'].extend(batch['charge'].cpu().numpy().tolist())
                dataframe['mass'].extend(batch['mass'].cpu().numpy().tolist())
                dataframe['peptide_length'].extend(batch['peplen'].cpu().numpy().tolist())
                dataframe['targ_intseq'].extend(batch['intseq'].cpu().numpy().tolist())
                dataframe['pred_intseq'].extend(prediction.cpu().numpy().tolist())
                dataframe['probs'].extend(predicted_probs.cpu().numpy().tolist())
                dataframe['targ_aaseq'].extend(targ_strings)
                dataframe['pred_aaseq'].extend(pred_strings)
                dataframe['correct_aa'].extend([result[0] for result in aa_matches_batch])
                dataframe['correct_peptide'].extend([result[1] for result in aa_matches_batch])
            
            # Naive metrics
            stats = U.AccRecPrec(target.cpu(), prediction.cpu(), self.head.NT)
            #vecs, auprc = U.RocCurve(target, prediction, probs, null_value=self.head.NT, typ='aa')
            #out['auprc'] += auprc
            
            # Add to totals
            for metric in dn_metrics['sum'].keys():
                if metric not in tots['sum'].keys():
                    tots['sum'][metric] = 0
                    tots['total'][metric] = 0
                tots['sum'][metric] += dn_metrics['sum'][metric]
                tots['total'][metric] += dn_metrics['total'][metric]

            for metric in stats.keys():
                metric_ = metric+'_naive'
                if metric_ not in tots['sum'].keys():
                    tots['sum'][metric_] = 0
                    tots['total'][metric_] = 0
                tots['sum'][metric_] += stats[metric]['sum']
                tots['total'][metric_] += stats[metric]['total']

            self.on_eval_step_end(target, loss_mask)
        
        steps = i+1
        totsz = self.config['loader']['batch_size']*steps
        out['ce'] = float((out['ce'] / (totsz * self.config['sl'])).cpu().detach().numpy())
        for metric in tots['sum'].keys():
            out[metric] = tots['sum'][metric] /  tots['total'][metric]

        self.on_eval_end()
        
        if save_df:
            return out, pd.DataFrame(dataframe)
        else:
            return out

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        highscore = 0
        for i in range(self.config['epochs']):
            
            # Train
            self.data.dataset['train'].set_epoch(i)
            self.train_epoch()
            self.on_train_epoch_end()
            
            # Eval
            out = self.evaluation(dset=eval_dset, max_batches=self.val_steps)
            
            # Logging
            if self.config['log_wandb']:
                out['epoch'] = i+1
                wandb.log(out)
                out.pop('epoch')
            
            specifier = " ".join(len(out)*['%s'])
            write_out = specifier%tuple([f"{m}={n:.3}" for m,n, in out.items()])
            line = "ValEpoch %d: %s"%(i, write_out)
            
            if out[self.config['high_score']] > highscore:
                highline = line
                highscore = out[self.config['high_score']]
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            print("\r"+line)
            
            # Saving the checkpoint
            if self.config['save_weights']:
                self.save_head(self.svdir+'weights/head_last.wts')
                self.save_encoder(self.svdir+'weights/encoder_last.wts')
                if highscore == out[self.config['high_score']]:
                    ext = f"epoch{i}_high_{highscore:.3f}"
                    wtsdir = os.path.join(self.svdir, "weights")
                    for file in glob(os.path.join(wtsdir, "*high*")): os.remove(file)
                    self.save_head(os.path.join(wtsdir, f"head_{ext}.wts"))
                    self.save_encoder(os.path.join(wtsdir, f"encoder_{ext}.wts"))
            
            self.eval_stats.append(list(out.values()))
            
            # Save data
            if self.log:
                self.savetxt(train_loss=None, eval_stats=np.array(self.eval_stats))
            
        return lines, highline

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_eval_step_end(self, *args, **kwargs):
        pass

    def on_eval_end(self, *args, **kwargs):
        pass


class DenovoArDSObj(BaseDenovo):
    def __init__(self, config, base_model=None, svdir='./dswts/'):
        task = 'denovo_ar'
        super().__init__(
            config=config, task=task, base_model=base_model, ar=True, 
            svdir=svdir
        )
        self.training_loss_keys = ['loss']

        # Head model
        head_dict = self.config[task]['head_dict']
        head_dict['kv_indim'] = self.encoder.run_units
        self.config['sl'] = self.config['loader']['pep_length'][1]
        self.head = DenovoDecoder(
            token_dict=self.data.amod_dic, dec_config=head_dict, 
            encoder=self.encoder # encoder is set by inherited class
        )
        self.predict_sequence = self.head.predict_sequence
        print(f"<DSCOMMENT> Total Decoder parameters: {self.head.decoder.total_params():,}")

        # loading previous weights
        if config['dswts'] is not None:
            regex = '*head*last*wts*' if config['load_last'] else "*head*wts*"
            print(f"<DSCOMMENT> Searching for decoder weights with regular expression {regex}")
            possible_weights_path = glob(os.path.join(self.svdir, "weights", regex))
            if len(possible_weights_path) > 1:
                try:
                    weights_path = [m for m in possible_weights_path if 'high' in m][0]
                    qualifier = '"high"'
                except:
                    weights_path = [m for m in glob(possible_weights_path) if 'last' in m][0]
                    qualifier = '"last"'
            else:
                weights_path = possible_weights_path[0]
                qualifier = 'only'
            print(f"<DSCOMMENT> Loading {qualifier} previous decoder weights")
            self.head.decoder.load_state_dict(th.load(weights_path, map_location=device))
        self.head.decoder.to(device)
        
        self.opt_head = th.optim.Adam(self.head.parameters(), self.starting_lr)
    
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
        
        return {'loss': loss}

    def log_wandb(self, losses, rlm, encoder_norm, decoder_norm):
        wandb.log({
            "Total loss": losses['loss'],
            "Total run loss": rlm['loss'],
            'Global step': self.global_step,
            "Global grad norm encoder": encoder_norm,
            "Global grad norm decoder": decoder_norm,
        })

from models.diffusion.model_utils import create_model_and_diffusion

class DenovoDiffusionObj(BaseDenovo):
    def __init__(self, config, diff_config=None, base_model=None, svdir='./dswts/'):
        task = 'denovo_diff'
        super().__init__(
            config=config, task=task, base_model=base_model, ar=False, 
            svdir=svdir
        )
        self.training_loss_keys = ['loss', 'mse', 'decoder_nll', 'tT']

        # Diffusion object
        if diff_config is None:
            with open("./yaml/diffusion.yaml") as stream:
                diff_config = yaml.safe_load(stream)
        if diff_config['learn_sigma']: self.training_loss_keys.append("vlb_terms")
        diff_config['pad_tok_id'] = self.data.amod_dic['X']
        diff_config['resume_checkpoint'] = False
        diff_config['sequence_len'] = self.config['loader']['pep_length'][1] + 1 # b/c of eos token
        self.diff_config = diff_config
        _, self.diff_obj = create_model_and_diffusion(**diff_config)

        # Head model
        head_dict = self.config[task]['head_dict']
        head_dict['kv_indim'] = self.encoder.run_units
        self.config['sl'] = self.config['loader']['pep_length'][1]
        self.head = DenovoDiffusionDecoder(
            input_output_units=diff_config['in_channel'],
            token_dict=self.data.amod_dic, 
            dec_config=head_dict,
            diff_obj=self.diff_obj,
            self_condition=config['denovo_diff']['self_condition'],
            clip_denoised=diff_config['clip_denoised'],
            output_sigma=diff_config['learn_sigma'],
            **config['denovo_diff']['head_dict'],
        )
        print(f"<DSCOMMENT> Total Decoder parameters: {self.head.total_params():,}")
        
        # loading previous weights
        if config['dswts'] is not None:
            regex = '*head*last*wts*' if config['load_last'] else "*head*wts*"
            print(f"<DSCOMMENT> Searching for decoder weights with regular expression {regex}")
            possible_weights_path = glob(os.path.join(self.svdir, "weights", regex))
            if len(possible_weights_path) > 1:
                try:
                    weights_path = [m for m in possible_weights_path if 'high' in m][0]
                    qualifier = '"high"'
                except:
                    weights_path = [m for m in glob(possible_weights_path) if 'last' in m][0]
                    qualifier = '"last"'
            else:
                weights_path = possible_weights_path[0]
                qualifier = 'last' if config['load_last'] else 'only'
            print(f"<DSCOMMENT> Loading {qualifier} previous decoder weights")
            self.head.load_state_dict(th.load(weights_path, map_location=device))
        
        self.head.to(device)
        self.opt_head = th.optim.Adam(self.head.parameters(), self.starting_lr)
        self.eval_score = []

    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])

        # Schedule sampler
        timesteps = th.empty(bs).uniform_(
            0, self.diff_obj.num_timesteps
        ).floor().type(th.int32).to(target.device)
        
        enc_input = self.encinp(batch, return_mask=True)

        target = self.append_null_token(target)
        target = self.replace_with_eos_token(target, batch['peplen'])

        loss_mask = self.head.sequence_mask(target)

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
        self.head.zero_grad()
        
        model_kwargs = {
            'input_ids': None,
            'decoder_input_ids': target,
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_feats': embedding['emb'],
        }
        if self.diff_config['use_loss_mask']:
            model_kwargs['loss_mask'] = loss_mask # THIS RUINS EVERYTHING

        losses = self.diff_obj.training_losses(
            self.head, 
            self.global_step,
            timesteps, 
            model_kwargs=model_kwargs, 
            noise=None
        )
        
        losses = {key: loss.mean() for key, loss in losses.items()}
        loss = losses['loss']
        loss.backward()
        
        if self.config['lr_warmup']:
            if self.global_step < self.config['lr_warmup_steps']:
                self.opt_head.param_groups[-1]['lr'] += self.lr_warmup_increment
                self.opt_encoder.param_groups[-1]['lr'] += self.lr_warmup_increment

        self.opt_head.step()
        if trenc:
            self.opt_encoder.step()
        
        return losses
    
    def log_wandb(self, losses, rlm, encoder_norm, decoder_norm):
        wandb.log({
            "Total loss": losses['loss'],
            "Total run loss": rlm['loss'],
            "MSE loss": losses['mse'],
            "MSE run loss": rlm['mse'],
            "DecoderNLL loss": losses['decoder_nll'],
            "DecoderNLL run loss": rlm['decoder_nll'],
            "tT loss": losses['tT'],
            "tT run loss": rlm['tT'],
            'Global step': self.global_step,
            "Global grad norm encoder": encoder_norm,
            "Global grad norm decoder": decoder_norm,
        })
        if 'vlb_terms' in losses:
            wandb.log({
                'VLB loss': losses['vlb_terms'],
                'VLB run loss': rlm['vlb_terms'],
            })
   
    def on_train_epoch_end(self):
        avg_losses = self.diff_obj.my_loss_history / (self.diff_obj.my_loss_count+1e-7)[...,None]
        save_path = os.path.join(self.svdir, "train_loss_by_timestep.tab")
        np.savetxt(save_path, avg_losses, delimiter='\t', fmt='%.8f')
        self.diff_obj.my_loss_history = np.zeros((self.diff_obj.num_timesteps, 3))
        self.diff_obj.my_loss_count = np.zeros((self.diff_obj.num_timesteps,))
    
    def on_eval_step_end(self, target, mask):
        pass
        #mses = np.zeros((target.shape[0], self.diff_obj.num_timesteps))
        #targ = self.head.get_embed(target)
        #for i, img in enumerate(self.diff_obj.my_xstart_save):
        #    mse = (mask[...,None]*(img-targ)).square().sum(dim=[1,2]) / mask.sum(-1) / img.shape[-1]
        #    mses[:, i] = mse.detach().cpu().numpy()
        #self.diff_obj.my_xstart_save = []
        #self.eval_score.append(mses)

    def on_eval_end(self):
        pass
        #ganz_batch = np.concatenate(self.eval_score, axis=0)
        #ganz_batch_mean = ganz_batch.mean(0)
        #self.eval_score = []
        #save_path = os.path.join(self.svdir, "eval_mse_by_timestep.tab")
        #np.savetxt(save_path, ganz_batch_mean, delimiter='\n', fmt='%.6f')

if __name__ == '__main__':

    # Read downstream yaml
    with open("./yaml/config.yaml") as stream:
        config = yaml.safe_load(stream)
    with open("./yaml/downstream.yaml") as stream:
        dsconfig = yaml.safe_load(stream)
        dsconfig['log'] = config['log']
        dsconfig['header'] = config['header']
    with open("./yaml/diffusion.yaml") as stream:
        diff_config = yaml.safe_load(stream)
    # Shorthand
    bs = dsconfig['batch_size']
    msg = dsconfig['log']
    swt = dsconfig['save_weights']
    # Overrides over a loaded previous experiment
    eval_only = dsconfig['eval_only']
    dswts = dsconfig['dswts']
    load_last = dsconfig['load_last']

    ########################################################
    # Create experiment directory in save/downstream_only/ #
    ########################################################

    # Continuing previous downstream run
    if dsconfig['dswts'] is not None:
        svdir = os.path.join(dsconfig['dswts'])
        with open(os.path.join(dsconfig['dswts'], "yaml", "downstream.yaml")) as stream:
            dsconfig = yaml.safe_load(stream)
        dsconfig['dswts'] = dswts
        dsconfig['load_last'] = load_last
    # Starting from pretrained encoder -> must fix to combine with create new exp
    elif dsconfig['pretrain_path'] is not None:
        svdir = os.path.join(dsconfig['pretrain_path'], 'weights')
    # Create new experiment
    elif (msg or swt):
        timestamp = U.timestamp()
        svdir = 'save/downstream_only/' + timestamp
        U.create_experiment(svdir, svwts=config['svwts'])
        with open(svdir + '/experiment_header', 'w') as f:
            f.write("Experiment header: " + config['header'])
        print("<DSCOMMENT> Experiment is writing to directory %s"%svdir)
    else:
        svdir = './'

    # Downstream object
    print("<DSCOMMENT> Denovo sequencing")
    if 'diff' in dsconfig['denovo_type']:
        print("<DSCOMMENT> Using diffusion decoder")
        D = DenovoDiffusionObj(dsconfig, svdir=svdir)
    else:
        print("<DSCOMMENT> Using autoregressive decoder")
        D = DenovoArDSObj(dsconfig, svdir=svdir)

    # WandB
    if dsconfig['log_wandb'] and (eval_only == False):
        wandb.init(
            project=config['project'],
            entity='joellapin',
			config={
				'master': config,
                'downstream': dsconfig,
                'diffusion': diff_config,
                'save_directory': svdir,
                'encoder_parameters': D.encoder.total_params(),
                'decoder_parameters': D.head.total_params(),
			},
		)   

    # Run training and/or evaluation
    if eval_only:
        out, df = D.evaluation(dset='test', max_batches=1e10, save_df=True)
        df.to_parquet(os.path.join(svdir, "output.parquet"))
        print("\n", out)
    else:
        print("Test validation", end='')
        out = D.evaluation(dset='val', max_batches=2)
        assert D.config['high_score'] in out.keys()
        print("\rTest validation passed")
        print(D.TrainEval()[-1])
