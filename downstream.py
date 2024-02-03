"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loaders.loader_parquet import LoaderDS
import numpy as np
from models.encoder import Encoder
from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.heads import SequenceHead, ClassifierHead
from models.decoder import DenovoDecoder
import os
from tqdm import tqdm
from collections import deque
from time import time
import utils as U
nn = th.nn
F = nn.functional
choice = np.random.choice
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class DownstreamObj:
    def __init__(self, config, task='denovo_ar', base_model=None):
        
        # Config is entire downstream yaml
        self.config = config
        self.task = task
        
        # Base model
        # - must do base model beforehand dataloader in order to transfer over 
        #   its configuration settings to self.config
        self.configure_encoder(base_model) # self.config updated
        #if not config['train_encoder']: self.encoder.trainable = False

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
            self.encoder.parameters(), self.config['lr']
        )

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

    def train_epoch(self, SeqInts=False):
        
        bs = self.config['batch_size']
        spe = self.config['steps_per_epoch']
        running_loss = deque(maxlen=50)
        
        T = tqdm(range(spe))
        perm = np.random.choice(self.dl.inds['train'], spe*bs, replace=True)
        for step in T:
            inds = perm[step*bs : (step+1)*bs]
            batch = self.dl.load_batch(inds, 'train', SeqInts=SeqInts)
        
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
            running_loss.append(loss.detach().cpu().numpy())
            
            if step%50==0:
                T.set_description(
                    "Loss: %.6f"%np.mean(running_loss), refresh=True
                )

class BaseDenovo(DownstreamObj):
    def __init__(self, 
                 config, 
                 task='denovo', 
                 base_model=None, 
                 ar=False, 
                 svdir='./dswts/'
                 ):
        super().__init__(config=config, task=task, base_model=base_model)
        self.ar = ar
        if svdir[-1] != '/': svdir += '/'
        if config['pretrain_path'] is not None:
            svdir = "/".join([config['pretrain_path'], svdir])
            if not os.path.exists(svdir):
                os.mkdir(svdir)

        self.svdir = svdir

    def evaluation(self, dset='val'):
        
        func = self.head.predict_sequence if self.ar else self.call
        
        # counters
        totsz = self.dl.dfs[dset].shape[0]
        steps = totsz // self.config['batch_size']
        steps += 0 if (totsz % self.config['batch_size'])==0 else 1
        
        # losses
        out = {'ce': 0, 'accuracy': 0, 'recall': 0, 'precision': 0}
        tots = {
            'accuracy': {'sum': 0, 'total': 0},
            'recall': {'sum': 0,'total': 0},
            'precision': {'sum': 0,'total': 0},
        }

        self.encoder.eval()
        self.head.eval()
        for step in tqdm(range(steps)):
            first = step*self.config['batch_size']
            last = np.minimum((step+1)*self.config['batch_size'], totsz)
            batch_inds = np.arange(first, last, 1)
            batch = self.dl.load_batch(batch_inds, dset=dset, SeqInts=True)
            batch = U.Dict2dev(batch, device)
            # Fork in the code for the 2 types of denovo models I created
            with th.no_grad():
                if self.ar:
                    enc_input, seqint, target = self.inptarg(
                        batch, full_seqint=True,
                    )
                    embedding = self.encoder(**enc_input)
                    prediction, probs = self.head.predict_sequence(embedding, batch)
                    pred = probs.transpose(-1,-2)
                else:
                    enc_input, target = self.inptarg(batch)
                    pred = func(enc_input) # logits
                    prediction = pred.argmax(-1).type(th.int32) # bs, sl
            
            out['ce'] += (
                F.cross_entropy(pred, target, reduction='none').sum()
                if self.ar else 
                self.LossFunction(target, pred).sum()
            )
            
            stats = U.AccRecPrec(target, prediction, null_value=self.dl.amod_dic['X'])
            for metric in stats.keys():
                for key, val in stats[metric].items():
                    tots[metric][key] += val
        
        out['ce'] = out['ce'] / (totsz * self.config['sl'])
        for metric in stats.keys():
            out[metric] = tots[metric]['sum'] / tots[metric]['total']

        return out

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        highscore = 0
        for i in range(self.config['epochs']):
            self.train_epoch(SeqInts=True) # Notice: SeqInts is true
            out = self.evaluation(dset=eval_dset)
            line = "ValEpoch %d: Cross-entropy=%.6f, Accuracy=%6f, Recall=%.6f, Precision=%.6f"%(
                (i,) + tuple(out.values())
            )
            if out['recall']>highscore:
                highline = line
                highscore = out['recall']
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            if self.config['save_weights']:
                self.save_head(self.svdir+'head.wts')
                if self.config['train_encoder']:
                    self.save_encoder(self.svdir+'encoder.wts')
            print(line)
        
        return lines, highline

class DenovoArDSObj(BaseDenovo):
    def __init__(self, config, base_model=None, svdir='./dswts/'):
        task = 'denovo_ar'
        super().__init__(
            config=config, task=task, base_model=base_model, ar=True, 
            svdir=svdir
        )

        # Dataloader
        self.dl = LoaderDS(self.config['loader'])
        self.predcats = len(self.dl.amod_dic)

        # Head model
        head_dict = self.config[task]['head_dict']
        head_dict['kv_indim'] = self.encoder.run_units
        self.config['sl'] = self.config['loader']['pep_length'][1]
        self.head = DenovoDecoder(
            token_dict=self.dl.amod_dic, dec_config=head_dict, 
            encoder=base_model
        )
        self.head.decoder.to(device)
        
        self.opt_head = th.optim.Adam(self.head.parameters(), config['lr'])
        

    def inptarg(self, batch, full_seqint=False):
        
        bs, sl = batch['seqint'].shape
        enc_input = self.encinp(batch, return_mask=True)
        
        # Take the variable batch['seqint'] and add a start token to the 
        # beginning and null on the end
        intseq = self.head.prepend_startok(batch['seqint'][...,:-1])
        
        # Find the indices first null tokens so that when you choose random
        # token you avoid trivial trailing null tokens (beyond final null)
        nonnull = (intseq != self.head.inpdict['X']).type(th.int32).sum(1)
        
        # Choose random tokens to predict
        # - the values of inds will be final non-hidden value in decoder input
        # - batch['seqint'](inds) will be the target for decoder output
        # - must use combination of rand() and round() because int32 is not
        #   yet implemented when feeding vectors into low/high arguments
        uniform = th.rand(bs, device=nonnull.device) * nonnull
        inds = uniform.floor().type(th.int32)
        
        # Fill with hidden tokens to the end
        # - this will be the decoder's input
        dec_inp = self.head.fill2c(intseq, inds, '<h>', output=False)
        
        # Indices of chosen predict tokens
        # - save for LossFunction
        inds_ = [th.arange(inds.shape[0], dtype=th.int32), inds]
        self.inds = inds_

        # Target is the actual (intseq) identity of the chosen predict indices
        targ = (
            batch['seqint'] if full_seqint else batch['seqint'][inds_]
        ).type(th.int64)

        return enc_input, dec_inp, targ

    def LossFunction(self, target, decout):
        targ_one_hot = F.one_hot(target, self.predcats).type(th.float32)
        logits = decout[self.inds]
        loss = F.cross_entropy(logits, targ_one_hot)

        return loss

    def train_step(self, batch, trenc=True):
        batch = U.Dict2dev(batch, device)
        enc_input, seqint, target = self.inptarg(batch)
        
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
        head_out = self.head(seqint, embedding, batch, training=True)
        all_loss = self.LossFunction(target, head_out)
        loss = all_loss.mean()
        
        loss.backward()
        self.opt_head.step()
        if trenc:
            self.opt_encoder.step()
        
        return loss

class DenovoBlDSObj(BaseDenovo):
    def __init__(self, config, base_model=None, svdir='./dswts/'):
        task='denovo_bl'
        super().__init__(
            config=config, task=task, base_model=base_model, svdir=svdir
        )

        head_dict = self.config[task]['head_dict']  
        # Dataloader
        self.dl = LoaderDS(self.config['loader'])
        
        # Head model
        # Place values into head dictionary that can't be determined beforehand 
        self.config['sl'] = self.config['loader']['pep_length'][1]# + 1
        head_dict['final_seq_len'] = self.config['sl']
        self.predcats = len(self.dl.amod_dic)
        head_dict['final_units'] = len(self.dl.amod_dic)
        self.head = SequenceHead(**head_dict)
        
        self.opt_head = th.optim.Adam(self.head.parameters(), config['lr'])

    def inptarg(self, batch):
        enc_input = self.encinp(batch, return_mask=True)
        target = batch['seqint'].type(th.int64)

        return enc_input, target

"""
# Read downstream yaml
with open("./yaml/downstream.yaml") as stream:
    config = yaml.safe_load(stream)

# Downstream object
print("Denovo sequencing")
D = DenovoArDSObj(config)
#out = D.evaluation(dset='val')
print("\n".join(D.TrainEval()))
"""
