################################################################################
#                                  todo                                        #
################################################################################
"""
Nada
"""
# dependencies used throughout the program
import os
import numpy as np
import torch as th
import yaml
import utils as U
import re
import wandb
from tqdm import tqdm
from glob import glob
import multiprocessing
multiprocessing.set_start_method('fork') # python 3.14 compatibility with dataloader
# slurm doesn't always manage gpus well -> cublas error
# you may need to set cuda_visible_devices={#} before python in shellscript.sh
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Device:", device)

################################################################################
#                        Configuration settings                                #
################################################################################


# Immediately read in yaml files to prevent unintended changes to the
# experiment while I am changing the files outside of the shellscript
with open("./yaml/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
with open("./yaml/datasets.yaml", 'r') as stream:
    dc = yaml.safe_load(stream)
with open("./yaml/models.yaml", 'r') as stream:
    mconf = yaml.safe_load(stream)
with open("./yaml/tasks.yaml", 'r') as stream:
    tc = yaml.safe_load(stream)
with open("./yaml/downstream.yaml") as stream:
    dsconfig = yaml.safe_load(stream)
with open(os.path.join("./denovo_base/yaml/config.yaml")) as stream:
    dnconfig = yaml.safe_load(stream)
config['lr'] = float(config['lr'])

# NOTE about trading information between yaml files:
# I don't want to have to specify inputs in 2 different yaml files that must be 
# consistent with each other. Instead, 1 yaml file can have all relevant input 
# specifications, and share its information with the configuration of the other.
# For example, specify bin size in yaml/tasks.yaml and use that value to set
# number of bins in hidden mz head model

# Header model
header_dict = mconf['header_dict']
# Set -ary tasks prediction classes
header_dict['tasks']['NaryTask']['num_classes'] = tc['NaryTask']['buckets']
header_dict['tasks']['VAE']['class_token_input'] = mconf['encoder_dict']['class_token']
header_dict['tasks']['VAE']['output_dim'] = int(np.floor((tc['VAE']['mzlims'][1] - tc['VAE']['mzlims'][0]) / tc['VAE']['binsz']))
#header_dict['tasks']['nary_ab']['num_classes'] = tc['nary_ab']['buckets']
#mzh = tc['hidden_mz']
#header_dict['tasks']['hidden_mz']['bins'] = int( 
#    (mzh['mzlims'][1] - mzh['mzlims'][0]) / mzh['binsz']
#) # hidden mz needs binsz and mz range apriori
#abh = tc['hidden_ab']
#header_dict['tasks']['hidden_ab']['bins'] = int(1 / abh['binsz']) # so does hidden ab
#header_dict['tasks']['hidden_spectrum']['bins'] = config['max_peaks']
# hidden charge needs max charge to set the number of classes
#header_dict['tasks']['hidden_charge']['num_classes'] = tc['hidden_charge']['max_charge']
header_dict = {task: header_dict['tasks'][task] for task in config['tasks']}
# pass encoder_dict's running_units to header_dict as in_units
header_dict['in_units'] = mconf['encoder_dict']['running_units']

dnconfig['encoder_dict'] = {**mconf['encoder_dict'], 'empty': False}
dnconfig['top_peaks'] = config['max_peaks']
dnconfig['batch_size'] = config['batch_size']
dnconfig['epochs'] = dsconfig['epochs']
print(f"Denovo runs will last {dnconfig['epochs']} epochs")
dnconfig['prev_wts'] = None
dnconfig['pretrained_encoder_path'] = None # loading encoder inside denovo base
dnconfig['loader']['val_name'] = dsconfig['loader']['val_species']
dnconfig['freeze_encoder'] = dsconfig['freeze_encoder']
dnconfig['save_weights'] = True if config['first_report']['only_dnv'] and config['svwts'] else False
dnconfig['log_wandb'] = True if config['first_report']['only_dnv'] else False
dnconfig['prev_wts'] = config['first_report']['loadpath'] if config['first_report']['only_dnv'] else None
for key in dsconfig:
    if 'lr_' in key: dnconfig[key] = dsconfig[key]

# Denovo downstream evaluation
if mconf['encoder_dict']['class_token']:
    config['nn_eval']['pooling'] = 'class_token'

################################################################################
#                                  Loader                                      #
################################################################################

from loaders.loader_hf import LoaderHF
from copy import deepcopy

L = LoaderHF(**dc['loader'])

################################################################################
#                                   Model                                      #
################################################################################

from models.encoder import Encoder
#from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.heads import Header
from utils import *

def turn_grad_on(encoder_model, grad_vector=None):
    if grad_vector == None:
        grad_vector = [True for m in encoder_model.parameters()]
    for parm, needs_grad in zip(encoder_model.parameters(), grad_vector):
        parm.requires_grad = needs_grad

def update_lr():
    warmup_left = config['lr_warmup_steps'] - lr_phase_count[0]
    if warmup_left > 0:
        step_size = (config['lr'] - optencoder.param_groups[-1]['lr']) / warmup_left
        optencoder.param_groups[-1]['lr'] += step_size
        for optimizer in header.opts.values():
            optimizer.param_groups[-1]['lr'] += step_size
        lr_phase_count[0] += 1
    else:
        lr_phase_count[1] += 1

# Encoder model
if mconf['encoder_name'] == 'depthcharge':
    print("Using Depthcharge encoder")
    encoder = dc_encoder(sequence_length=config['max_peaks'])
else:
    print("Using user encoder")
    encoder_dic = mconf['encoder_dict']
    encoder = Encoder(**encoder_dic)
encoder.to(device) # model shouldn't need to come off of GPU entire run
print(f"Total encoder parameters: {encoder.total_params():,}")

# Header model(s)
header = Header(header_dict, lr=1e-7)
for task in header.heads.keys(): header.heads[task].to(device)
assert hasattr(header, 'name')

# Optimizers
lr_phase_count = [0,0]
optencoder = th.optim.Adam(encoder.parameters(), 1e-7)

if config['loadpath'] is not None:
    print("Loading previous experiment: ", end="")
    # Encoder
    loadpath = os.path.join(config['loadpath'], 'weights')
    enc_file_name = U.find_file('model_enc', loadpath)
    result = encoder.load_state_dict(th.load(enc_file_name, map_location=device))
    print(result)
    opt_file_name = U.find_file('opt_encopt', loadpath)
    U.load_optimizer_state(optencoder, opt_file_name, device)
    
    # Head(s)
    for task in config['tasks']:
        try:
            head_file_name = U.find_file(task, loadpath)
            header.heads[task].load_state_dict(th.load(head_file_name, map_location=device))
            # ASSUMPTION: header optimizers follow name convention 
            # opt_{task}.wts.npy
            opt_file_name = U.find_file("opt_%s"%task, loadpath)
            U.load_optimizer_state(
                header.opts[task], 
                opt_file_name,
                device
            )
        except:
            print(f"Weights for {task} task not found")

    save_path = config['loadpath'] if dnconfig['save_weights']==False else None
else:
    save_path = None

################################################################################
#                                    Loss                                      #
################################################################################

import tasks
import importlib
import inspect
import pkgutil
classes_dict = {}
# Iterate over all modules inside the package directory
for _, module_name, is_pkg in pkgutil.iter_modules(tasks.__path__):
    if not is_pkg:
        # Dynamically import the module
        full_module_name = f"{tasks.__name__}.{module_name}"
        module = importlib.import_module(full_module_name)
        # Extract all classes defined directly within that module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == full_module_name:
                classes_dict[name] = obj

T = {}
for task in config['tasks']:
    T_ = classes_dict[task]
    T[task] = T_(**tc[task])

# All tasks have loss variables for tracking
loss_spec = " ".join(['%s: %%7.5f'%task_name for task_name in T.keys()])

################################################################################
#                           Downstream evaluation                              #
################################################################################

################################################################################
#                    denovo_base evaluation (AR sequencing)                    #
################################################################################

def denovo_base_eval(encoder, svdir='./denovo_eval/', freeze_encoder=True):
    """
    Snapshot `encoder`'s current weights into denovo_base's DenovoArObj
    (autoregressive de novo sequencing model) and train/evaluate it for 1
    epoch, as a sanity check on the encoder currently being pretrained.

    denovo_base (./denovo_base/) is a separate, standalone repo that expects
    to own the bare `models`/`utils` module names when it imports itself.
    Since this script already has its own `models`/`utils` loaded under those
    same names, they're stashed out of sys.modules for the duration of the
    import so denovo_base's copies don't clobber them.
    """
    import sys
    import importlib
    import gc
    
    needs_grad = [m.requires_grad for m in encoder.parameters()]
    root = os.path.dirname(os.path.abspath(__file__))
    denovo_base_dir = os.path.join(root, "denovo_base")
    collide = lambda name: name == 'models' or name.startswith('models.') or name == 'utils'

    stashed = {name: sys.modules.pop(name) for name in list(sys.modules) if collide(name)}
    sys.path.insert(0, denovo_base_dir)
    try:
        model_runners = importlib.import_module("denovo_base.models.model_runners")
        dnconfig['freeze_encoder'] = dnconfig['freeze_encoder']#freeze_encoder
        rddir = svdir if dnconfig['prev_wts'] else None
        DS = model_runners.DenovoArObj(dnconfig, svdir=svdir, rddir=rddir, encoder_model=encoder)
        # Insert a snapshot of the current encoder's weights (a copy, so this
        # evaluation can't perturb the encoder actually being pretrained)
        DS.model.encoder.load_state_dict(encoder.state_dict())

        out = DS.TrainEval()
    finally:
        sys.path.remove(denovo_base_dir)
        for name in [n for n in sys.modules if collide(n)]:
            del sys.modules[name]
        sys.modules.update(stashed)
        del DS.data.dataloader['train']
        del DS.data.dataloader['test']
        gc.collect()
        turn_grad_on(encoder, grad_vector=needs_grad)

    #print("<PMCOMMENT> denovo_base 1-epoch evaluation:")
    #print("\n".join(lines))
    return out[-1]

################################################################################
#                    nearest-neighbor evaluation (encoder embeddings)          #
################################################################################

import nn_eval

def nearest_neighbor_eval(encoder, svdir='./', dset='val', write_every=False):
    """
    Encode the held-out test set with `encoder`'s current weights and, for
    each spectrum, write its top-n nearest neighbors (by pooled embedding
    similarity) to a parquet file under `svdir`/nn_eval/.
    """
    cfg = config['nn_eval']
    outdir = os.path.join(svdir, 'nn_eval')
    os.makedirs(outdir, exist_ok=True)
    if not write_every:
        for file in glob(os.path.join(outdir, "*.parquet")): os.remove(file)
    output_path = os.path.join(outdir, "step_%d.parquet" % encoder.global_step.item())

    out = nn_eval.evaluate_similarity(
        encoder, L, output_path,
        top_n=cfg['top_n'], pooling=cfg['pooling'], metric=cfg['metric'],
        query_batch_size=cfg['query_batch_size'],
    )
    if config['log_wandb']: wandb.log({'nn_frac': np.mean(out)})

    return output_path

################################################################################
#                                  Training                                    #
################################################################################

from collections import deque
from time import time
import sys
import datetime

def train_step(batch, task, enc_opt, head_opt):
    # Verify training mode, device, and zero grads
    encoder.train()
    encoder.zero_grad()
    enc_opt.zero_grad()
    header.train()
    header.zero_grad()
    head_opt.zero_grad()
    batch = U.Dict2dev(batch, device, inplace=False)
    head_outputs = [task]

    inp = T[task].inptarg(batch)
    inp['length'] = batch['length']
    
    if hasattr(encoder, 'its'):
        if encoder.its == 1: 
            enc_output = encoder(**inp)
        else:
            enc_output = encoder.RecycleTrainOutput(inp)
    else:
        enc_output = encoder(**inp)

    prediction = header(enc_output['emb'], head_outputs)
    loss = T[task].loss(prediction[task])
    loss = loss.mean()
    
    loss.backward()
    enc_opt.step()
    head_opt.step()

    update_lr()
    
    encoder.global_step +=1

    return loss

def save_train_loss(filepath, loss_list):
    if os.path.exists(filepath):
        loss_list = np.append(np.loadtxt(filepath), np.array(loss_list))
    np.savetxt(filepath, loss_list, fmt='%.6f')

def train(epochs=1, runlen=50, svfreq=3600, save_path=None):
    
    # Shorthand
    bs = config['batch_size']
    swt = config['svwts'] & (config['debug']!=True)
    
    # Create experiment directory in save/
    timestamp = U.timestamp()
    if config['first_report']['only_dnv'] & (config['first_report']['loadpath']!=None):
        svdir = config['first_report']['loadpath']
    elif swt:
        if save_path is None:
            svdir = 'save/' + timestamp
            U.create_experiment(svdir, svwts=config['svwts'])
        else:
            timestamp = save_path.split('/')[-1]
            svdir = save_path
        #if config['svwts']: 
        #    U.save_all_weights(svdir, (encoder, optencoder), header, remark="step_0_loss_999999999")
    else:
        svdir = './' # for establishing ds objects below
    
    # Variables needed for saving gradient infomration
    if config['svgrad']:
        parms_enc = [str(tuple(parm.shape)) for parm in encoder.parameters()]
        parms_head = [
            str(tuple(parm.shape)) for parm in header.heads['nary_mz'].parameters()
        ]
        parmshapes = parms_enc + parms_head
        parmgrads = []
        all_loss = []

    if config['log_wandb']:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            config={
                'config': {'main': config, 'datasets': dc, 'models': mconf, 'tasks': tc, 'downstream': dnconfig},
                'save_directory': timestamp,
                'encoder_parameters': encoder.total_params(),
            },
        )
    sys.stdout.write("Starting training for %d epochs\n"%epochs)

    # First report
    if config['first_report']['pretrain_execute'] | config['first_report']['only_dnv']:
        if config['dnv_eval']['execute'] | config['first_report']['only_dnv']:
            eval_out = denovo_base_eval(encoder, svdir=svdir)
            if config['first_report']['only_dnv']: sys.exit()
            eval_out = dict(zip(['aa_recall', 'peptide'], map(eval_out.get, ['aa_recall', 'peptide'])))
            if config['log_wandb']:
                wandb.log({'global_step': encoder.global_step.item()} | eval_out)
        if config['nn_eval']['execute'] and 'val' in L.dataloader:
            nearest_neighbor_eval(encoder, svdir=svdir)

    # Train
    svtime = time()

    eval_loss = 0
    loss_list = []
    max_steps_tick=False
    for epoch in range(epochs):
        start_epoch = time()
        for task_name, task in T.items(): task.reset_total_loss()
        
        pbar = tqdm(
            L.dataloader['train'],
            #total=train_steps,
            smoothing=0.6,
            #disable=not self.accelerator.is_local_main_process
        )
        L.dataset['train'].set_epoch(epoch)
        start_load = time()
        for step, batch in enumerate(pbar):
            start_step = time()
            
            # Train model for a step
            random_task = np.random.choice(list(T.keys()), 1)[0]
            loss = train_step(
                batch, random_task, optencoder, header.opts[random_task]
            )
            
            # Save running stats
            loss = loss.item()
            T[random_task].log_loss(loss)
            if config['log_wandb']:
                wandb.log({
                    'learning_rate': optencoder.param_groups[-1]['lr'],
                    'global_step': encoder.global_step.item(), 
                    random_task: loss,
                })
            
            # Gradient tracking
            if config['svgrad']:
                grads = (
                    [float(parm.grad.norm().detach().cpu().numpy()) 
                     for parm in encoder.parameters() if parm.requires_grad] +
                    [float(parm.grad.norm().detach().cpu().numpy()) 
                     for parm in header.heads['trinary_mz'].parameters() if parm.requires_grad]
                )
                parmgrads.append(grads)
                all_loss.append(T[random_task].running_loss['main'][-1])
            
            # Stdout
            if step%10==0:
                means = tuple([task.calc_avg_running_loss()['main'] for task_name, task in T.items()])
                loss_string = loss_spec%means
                pbar.set_description(f"\rStep {step}, loss={loss_string}")
            
            # Saving weights and testing
            if config['svwts'] and (time()-svtime > svfreq):
                remark = f"step_{encoder.global_step.item()}_last"
                U.save_all_weights(svdir, (encoder, optencoder), header, remark=remark, clear=True)
                #remark = "step_%d_loss_%.5f"%(encoder.global_step.item(), eval_loss)
                #last_loss = float(".".join(U.find_file("model_enc", svdir+'/weights').split('_')[-1].split('.')[:-1]))
                #if swt & (eval_loss < last_loss):
                #    U.save_all_weights(svdir, (encoder, optencoder), header, remark=remark, clear=True)
                svtime = time()

            # Run evaluation and save training_loss
            if encoder.global_step % config['steps_per_report'] == 0:
                if config['dnv_eval']['execute']:
                    eval_out = denovo_base_eval(encoder)
                    eval_out = dict(zip(['aa_recall', 'peptide'], map(eval_out.get, ['aa_recall', 'peptide'])))
                    sys.stdout.write(f"\rEvaluation @ Global step={encoder.global_step.item()}: aa={eval_out['aa_recall']}, peptide={eval_out['peptide']}\n")
                    if config['log_wandb']:
                        wandb.log({'global_step': encoder.global_step.item()} | eval_out)
                if config['nn_eval']['execute']:
                    nn_path = nearest_neighbor_eval(encoder, svdir=svdir)
                    sys.stdout.write(f"\rWrote nearest-neighbor eval to {nn_path}\n")
                loss_list = []
                
            start_load = time()
            
            # Arrest training at max_steps
            if int(encoder.global_step) == config['max_steps']:
                print()
                max_steps_tick = True
                break

        # End of epoch
        if max_steps_tick:
            break

        # Std out and logging
        tot_losses = tuple([
            task.calc_avg_total_loss()['main'] for task_name, task in T.items()
        ])
        loss_string = loss_spec%tot_losses
        Line = 'Epoch %d, Global step %d: mean_loss=%s (%d s)'%(
            epoch, encoder.global_step.cpu().detach().numpy(), loss_string, time()-start_epoch
        )
        sys.stdout.write("\r\033[K%s\n"%Line)

    # End of pre-training
    # Save weights, perhaps
    #if swt:
    #    U.save_all_weights(svdir, (encoder, optencoder), header, remark='final')
    # Save gradients, perhaps
    if config['svgrad']:
        with open(svdir+"/parmshapes", 'w') as f: 
            f.write("|".join(parmshapes))
        np.savetxt(svdir+"/allloss", np.array(all_loss))
        np.savetxt(svdir+"/parmgrads", np.array(parmgrads))

        
    print()

if __name__ == '__main__':
    
    if config['seed'] is not None:
        np.random.seed(config['seed'])
        th.manual_seed(config['seed'])
    train(epochs=config['epochs'], runlen=100, svfreq=config['svfreq'], save_path=save_path)

