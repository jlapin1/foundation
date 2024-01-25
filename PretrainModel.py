###############################################################################
#                                  todo                                       #
###############################################################################
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
Adam = th.optim.Adam
# slurm doesn't always manage gpus well -> cublas error
# you may need to set cuda_visible_devices={#} before python in shellscript.sh
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Device:", device)

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
config['lr'] = float(config['lr'])

clip_op = (
    th.nn.utils.clip_grad_value_
    if config['headclip']['type'] == 'value' else
    th.nn.utils.clip_grad_norm_
)

# NOTE about trading information between yaml files:
# I don't want to have to specify inputs in 2 different yaml files that must be 
# consistent with each other. Instead, 1 yaml file can have all relevant input 
# specifications, and share its information with the configuration of the other.
# For example, specify bin size in yaml/tasks.yaml and use that value to set
# number of bins in hidden mz head model

# Header model
header_dict = mconf['header_dict']
mzh = tc['hidden_mz']
header_dict['tasks']['hidden_mz']['bins'] = int( 
    (mzh['mzlims'][1] - mzh['mzlims'][0]) / mzh['binsz']
) # hidden mz needs binsz and mz range apriori
abh = tc['hidden_ab']
header_dict['tasks']['hidden_ab']['bins'] = int(1 / abh['binsz']) # so does hidden ab
header_dict['tasks']['hidden_spectrum']['bins'] = config['max_peaks']
# hidden charge needs max charge to set the number of classes
header_dict['tasks']['hidden_charge']['num_classes'] = tc['hidden_charge']['max_charge']
header_dict = {task: header_dict['tasks'][task] for task in config['tasks']}
# pass encoder_dict's running_units to header_dict as in_units
header_dict['in_units'] = mconf['encoder_dict']['running_units']
# Override downstream saving if log is False
if config['svwts'] is False:
    dsconfig['save_weights'] = False
# Override/add to downstream/dataset top_pks
dsconfig['loader']['top_pks'] = config['max_peaks']
dc['pretrain']['top_pks'] = config['max_peaks']
# set downstream encoder_dict
dsconfig['encoder_dict'] = mconf['encoder_dict']
# set kv_indim in decoder_dict to the enocoder's running_units
dsconfig['denovo_ar']['head_dict']['running_units'] = mconf['encoder_dict']['running_units']



###############################################################################
#                                  Loader                                     #
###############################################################################

from loaders.loader import DatasetObj, DataLoader
from copy import deepcopy

dataset = DatasetObj(**dc['pretrain'])
L = DataLoader(
    dataset=dataset,
    num_workers=dc['num_workers'],
    batch_size=config['batch_size'],
    shuffle=True,
)

###############################################################################
#                                   Model                                     #
###############################################################################

from models.encoder import Encoder
from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.heads import Header
from utils import *

# Encoder model
if mconf['encoder_name'] == 'depthcharge':
    print("Using Depthcharge encoder")
    encoder = dc_encoder(sequence_length=config['max_peaks'])
else:
    print("Using user encoder")
    encoder_dic = mconf['encoder_dict']
    encoder = Encoder(**encoder_dic)
encoder.to(device) # model shouldn't need to come off of GPU entire run
print("Total encoder parameters: %d"%encoder.total_params())

# Header model(s)
header = Header(header_dict)#, lr=config['lr'])
for task in header.heads.keys(): header.heads[task].to(device)
assert hasattr(header, 'name')

# Optimizers
optencoder = Adam(encoder.parameters(), config['lr'])

def save_all_weights(svdir):
    U.save_full_model(encoder, optencoder, svdir)
    # Save all header weights in one file
    th.save(header.state_dict(), "save/%s/weights/model_%s.wts"%(svdir, header.name))
    # Save header optimizer weights individually
    for task_name in config['tasks']:
        # optimizer.name should have opt_ already in it (see Header in models)
        fn = 'save/%s/weights/opt_%s.wts'%(svdir, task_name)
        U.save_optimizer_state(header.opts[task_name], fn)

if config['load']:
    ldpth = config['loadpath']
    model.load_state_dict(th.load(ldpth + 'model_encoder.wts'))
    U.load_optimizer_state(
        optencoder, ldpth + 'opt_encopt.wts', device
    )
    header.load_state_dict(th.load(ldpth + 'model_header.wts'))
    for task_name in config['tasks']:
        # ASSUMPTION: header optimizers follow name convention 
        # opt_{task}.wts.npy
        U.load_optimizer_state(
            header.opts[task_name], 
            ldpth + 'opt_%s.wts'%task_name,
            device
        )

###############################################################################
#                                    Loss                                     #
###############################################################################

import tasks

# All tasks have loss variables for tracking
T = tasks.all_tasks(tc)
T = {task: T[task] for task in config['tasks']}
loss_spec = " ".join(['%s: %%7.5f'%task_name for task_name in T.keys()])

###############################################################################
#                           Downstream evaluation                             #
###############################################################################

if not config['debug']:
    import downstream as ds

    # Downstream object
    allds = {
        #'charge': ds.ChargeDSObj,
        #'peplen': ds.PeplenDSObj,
        'denovo_ar': ds.DenovoArDSObj,
        #'denovo_bl': ds.DenovoBlDSObj,
    }

###############################################################################
#                                  Training                                   #
###############################################################################

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

    inp = T[task].inptarg(batch)
    inp['length'] = batch['length']

    head_outputs = [task]
    enc_output = encoder(**inp)
    prediction = header(enc_output['emb'], head_outputs)
    loss = T[task].loss(prediction[task])
    loss = loss.mean()
    
    loss.backward()
    enc_opt.step()
    if config['headclip']['use']: 
        parms = header.heads[task].parameters()
        clip_op(parms, config['headclip']['max'])
    head_opt.step()
    
    encoder.global_step +=1 

    return loss

def evaluation(steps=100):
    lst = [
        'Qm', 'Qs', 'Km', 'Ks', 
        'Vm', 'Vs', 'QKm', 'QKs', 
        'WTSm', 'WTSs', 'ATTm', 'ATTs', 
        'RESATTm', 'RESATTs', 'FFN1m', 
        'FFN1s', 'FFN2m', 'FFN2s', 'TBm', 'TBs'
    ]
    ll = len(lst)

    encoder.eval()
    with th.no_grad():
        others = np.zeros((9,ll))
        for step, batch in enumerate(L):
            if step == steps: break
            print("\rEvaluation step %d/%d"%(step, steps), end='')
            
            batch = U.Dict2dev(batch, device, inplace=False)

            mzab_inp = th.cat([batch['mz'][...,None], batch['ab'][...,None]], -1)
            inp = {
                'x': mzab_inp,
                'charge': None,
                'mass': None,
                #'length': batch['length'],
                'return_mask': True,
                'return_full': True
            }
            enc_output = encoder(**inp)
            stats = np.stack([
                np.concatenate([[n.cpu().detach().numpy().mean(), n.cpu().detach().numpy().std()] for n in line]) 
                for line in enc_output['other']
            ])
            others += stats
    others /= steps

    with open("activations.txt", "a") as f:
        f.write((" ".join(ll*['%8s']))%tuple(lst) + '\n')
        for m in range(9):
            f.write((" ".join(ll*["%8.5f"]))%tuple(others[m]) + '\n')

def train(epochs=1, runlen=50, svfreq=3600):
    
    # Shorthand
    bs = config['batch_size']
    msg = config['log'] & (config['debug']!=True)
    swt = config['svwts'] & (config['debug']!=True)
    # Create experiment directory in save/
    if (msg or swt):
        dt = str(datetime.datetime.now()).split()
        dt[-1] = re.sub(':','-',dt[-1]) # linux has issue with : symbol
        svdir = "_".join(dt)
        os.mkdir('save/%s'%svdir);
        os.mkdir('save/%s/yaml'%svdir)
        os.system("cp ./yaml/*.yaml save/%s/yaml/"%svdir)
        if config['svwts']: 
            os.mkdir('save/%s/weights'%svdir)
            save_all_weights(svdir)
    else:
        svdir = './' # for establishing ds objects below
    
    # Log starting messages and start collection all lines
    if msg:
        line = "%s\nTotal parameters: %d\n"%(
            config['header'], encoder.total_params()
        )
        U.message_board(line, "save/%s/epochout.txt"%svdir)
        #U.message_board(line, "save/%s/valout.txt"%svdir)
        line = "%s\n%s\n"%(svdir, config['header'])
        allepochlines = [line]
        #allvallines = [line]

    # Variables needed for saving gradient infomration
    if config['svgrad']:
        parms_enc = [str(tuple(parm.shape)) for parm in encoder.parameters()]
        parms_head = [
            str(tuple(parm.shape)) for parm in header.heads['trinary_mz'].parameters()
        ]
        parmshapes = parms_enc + parms_head
        parmgrads = []
        all_loss = []
    
    # Train
    running_time = deque(maxlen=runlen) # Full time
    load_time = deque(maxlen=runlen) # load_batch time
    graph_time = deque(maxlen=runlen) # train_step time
    svtime = time()
    sys.stdout.write("Starting training for %d epochs\n"%epochs)
    
    if config['eval_steps']>0: evaluation(config['eval_steps'])
    for epoch in range(epochs):
        start_epoch = time()
        for task_name, task in T.items(): task.reset_total_loss()
        
        start_load = time()
        for step, batch in enumerate(L):
            running_time.append(0 if step==0 else time()-start_step)
            start_step = time()
            load_time.append(start_step-start_load)
            
            # Train model for a step
            random_task = np.random.choice(list(header.heads.keys()), 1)[0]
            
            TT=time();loss = train_step(
                batch, random_task, optencoder, header.opts[random_task]
            );graph_time.append(time()-TT) # train_step time sandwich
            
            # Save running stats
            T[random_task].log_loss(loss.detach().cpu().numpy())
            running_time.append(time()-start_step)
            
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
            if step%50==0:
                means = tuple([
                    task.calc_avg_running_loss()['main']
                    for task_name, task in T.items()
                ])
                loss_string = loss_spec%means
                sys.stdout.write(
                    "\r\033[KStep %6d/%6d, loss=%s (%.2f,%.2f,%.2f s)"%(
                        step, config['steps_per_epoch'], loss_string, 
                        np.mean(running_time), np.mean(load_time), 
                        np.mean(graph_time)
                    )
                )
            
            # Saving weights and testing
            if time()-svtime > svfreq:
                if swt:
                    save_all_weights(svdir)
                svtime = time()

            if step == config['steps_per_epoch']-1:
                break

            start_load = time()

        # End of epoch
        tot_losses = tuple([
            task.calc_avg_total_loss()['main'] for task_name, task in T.items()
        ])
        loss_string = loss_spec%tot_losses
        Line = 'Epoch %d, Global step %d: mean_loss=%s (%d s)'%(
            epoch, encoder.global_step.cpu().detach().numpy(), loss_string, time()-start_epoch
        )
        sys.stdout.write("\r\033[K%s\n"%Line)
        if msg:
            U.message_board(Line+'\n', "save/%s/epochout.txt"%svdir)
            allepochlines.append(Line+"\n")

        if config['eval_steps']>0: evaluation(config['eval_steps'])
    
    # End of pre-training
    # Save weights, perhaps
    if swt:
        save_all_weights(svdir)
    # Save gradients, perhaps
    if config['svgrad']:
        with open('save/'+svdir+"/parmshapes", 'w') as f: 
            f.write("|".join(parmshapes))
        np.savetxt('save/'+svdir+"/allloss", np.array(all_loss))
        np.savetxt('save/'+svdir+"/parmgrads", np.array(parmgrads))

    # Run quick(ish) few shot downstream evaluation
    if config['downstream'] is not None:
        for dstask in config['downstream']:
            DS = allds[dstask](
                dsconfig, base_model=encoder, 
                svdir='save/%s/dswts/%s/'%(svdir, dstask)
            )
            sys.stdout.write("\r\033[KDownstream evlauation: %s\n"%(dstask))
            line, highline = DS.TrainEval()
            Line = "Downstream evlauation: %s; "%(dstask) + highline
            sys.stdout.write("\r\033[K%s\n"%Line)
            if msg:
                U.message_board("\n".join(line)+'\n', "save/%s/epochout.txt"%svdir)
                allepochlines.append(Line+"\n")
    if msg:
        # Append results to the .all files
        U.message_board("".join(allepochlines), "save/epochout.all")
    
    print()

if config['seed'] is not None:
    np.random.seed(config['seed'])
    th.manual_seed(config['seed'])
train(epochs=config['epochs'], runlen=100, svfreq=config['svfreq'])

