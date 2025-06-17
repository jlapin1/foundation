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
Adam = th.optim.Adam
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
config['lr'] = float(config['lr'])

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
dsconfig['loader']['batch_size'] = config['batch_size']
dc['loader']['top_pks'] = config['max_peaks']
dc['loader']['batch_size'] = config['batch_size']
# set downstream encoder_dict
dsconfig['encoder_dict'] = mconf['encoder_dict']
# set kv_indim in decoder_dict to the enocoder's running_units
dsconfig['denovo_ar']['head_dict']['running_units'] = mconf['encoder_dict']['running_units']
# Log downstream if logging pretraining
dsconfig['log'] = config['log']
dsconfig['header'] = config['header']

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
header = Header(header_dict)#, lr=config['lr'])
for task in header.heads.keys(): header.heads[task].to(device)
assert hasattr(header, 'name')

# Optimizers
optencoder = Adam(encoder.parameters(), config['lr'])

if config['load']:
    # Encoder
    ldpth = config['loadpath']
    enc_file_name = U.find_file('model_enc', ldpth)
    encoder.load_state_dict(th.load(enc_file_name, map_location=device))
    opt_file_name = U.find_file('opt_encopt', ldpth)
    U.load_optimizer_state(optencoder, opt_file_name, device)
    
    # Head(s)
    for task in config['tasks']:
        head_file_name = U.find_file(task, ldpth)
        header.heads[task].load_state_dict(th.load(head_file_name, map_location=device))
        # ASSUMPTION: header optimizers follow name convention 
        # opt_{task}.wts.npy
        opt_file_name = U.find_file("opt_%s"%task, ldpth)
        U.load_optimizer_state(
            header.opts[task], 
            opt_file_name,
            device
        )

################################################################################
#                                    Loss                                      #
################################################################################

import tasks

# All tasks have loss variables for tracking
T = tasks.all_tasks(tc)
T = {task: T[task] for task in config['tasks']}
loss_spec = " ".join(['%s: %%7.5f'%task_name for task_name in T.keys()])

################################################################################
#                           Downstream evaluation                              #
################################################################################

if (config['downstream'] is not None) and (not config['debug']):
    import downstream as ds

    # Downstream object
    allds = {
        'denovo_ar': ds.DenovoArDSObj,
    }

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
    
    encoder.global_step +=1 

    return loss

def evaluation(task):
    encoder.eval()
    header.eval()

    tot = 0
    count = 0
    with th.no_grad():
        for step, batch in enumerate(L.dataloader['val']):
            print("\rEvaluation step %d%50s"%(step+1, ""), end='')
            batch = U.Dict2dev(batch, device, inplace=False)
            inp = T[task].inptarg(batch)
            inp['length'] = batch['length']
            head_outputs = [task]
            
            enc_output = encoder(**inp)
            prediction = header(enc_output['emb'], head_outputs)
            loss = T[task].loss(prediction[task])

            tot += loss.sum()
            count += np.prod(tuple(loss.shape))
        mean_loss = float(tot.detach().cpu().numpy()) / count
    print("\rValidation loss at step %d: %.6f%50s"%(encoder.global_step, mean_loss, ""))

    return mean_loss

def save_train_loss(filepath, loss_list):
    if os.path.exists(filepath):
        loss_list = np.append(np.loadtxt(filepath), np.array(loss_list))
    np.savetxt(filepath, loss_list, fmt='%.6f')

def train(epochs=1, runlen=50, svfreq=3600):
    
    # Shorthand
    bs = config['batch_size']
    msg = config['log'] & (config['debug']!=True)
    swt = config['svwts'] & (config['debug']!=True)
    
    # Create experiment directory in save/
    if (msg or swt):
        timestamp = U.timestamp()
        svdir = 'save/' + timestamp
        U.create_experiment(svdir, svwts=config['svwts'])
        if config['svwts']: 
            U.save_all_weights(svdir, (encoder, optencoder), header, remark="step_0_loss_999999999")
    else:
        svdir = './' # for establishing ds objects below
    
    # Log starting messages and start collection all lines
    if msg:
        line = f"Experiment header: {config['header']}\nTotal parameters: {encoder.total_params():,}\n"
        U.message_board(line, "%s/epochout.txt"%svdir)
        line = "%s\n%s\n"%(timestamp, config['header'])
        allepochlines = [line]

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
    
    eval_loss = 999999999
    loss_list = []
    max_steps_tick=False
    for epoch in range(epochs):
        start_epoch = time()
        for task_name, task in T.items(): task.reset_total_loss()
        
        L.dataset['train'].set_epoch(epoch)
        start_load = time()
        for step, batch in enumerate(L.dataloader['train']):
            running_time.append(0 if step==0 else time()-start_step)
            start_step = time()
            load_time.append(start_step-start_load)
            
            # Train model for a step
            TT=time()
            #T['trinary_mz'].stdev = 0.5*np.exp(-6.9314718055994526e-06*float(encoder.global_step))
            random_task = np.random.choice(list(header.heads.keys()), 1)[0]
            loss = train_step(
                batch, random_task, optencoder, header.opts[random_task]
            )
            
            # Save running stats
            loss = loss.detach().cpu().numpy()
            T[random_task].log_loss(loss)
            loss_list.append(T[random_task].calc_avg_running_loss()['main'])
            running_time.append(time()-start_step)
            graph_time.append(time()-TT)
            
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
                means = tuple([
                    task.calc_avg_running_loss()['main']
                    for task_name, task in T.items()
                ])
                loss_string = loss_spec%means
                sys.stdout.write(
                    "\r\033[KStep %6d, loss=%s (%.3f,%.3f,%.3f s)"%(
                        step, loss_string, 
                        np.mean(running_time), np.mean(load_time), 
                        np.mean(graph_time)
                    )
                )
            
            # Saving weights and testing
            if time()-svtime > svfreq:
                remark = "step_%d_loss_%.5f"%(encoder.global_step, eval_loss)
                last_loss = float(".".join(U.find_file("model_enc", svdir+'/weights').split('_')[-1].split('.')[:-1]))
                if swt & (eval_loss < last_loss):
                    U.save_all_weights(svdir, (encoder, optencoder), header, remark=remark, clear=True)
                svtime = time()

            # Run evaluation and save training_loss
            if (step+1) % config['steps_per_report'] == 0:
                eval_loss = evaluation(list(T.keys())[0])
                if msg:
                    Line = "Validation loss at step %d: %.6f\n"%(step+1, eval_loss)
                    U.message_board(Line, "%s/epochout.txt"%svdir)
                    save_train_loss("%s/train_loss.txt"%svdir, loss_list)
                    allepochlines.append(Line)
                loss_list = []
                
            start_load = time()
            
            # Arrest training at max_steps
            if int(encoder.global_step) == config['max_steps']:
                print()
                if msg: save_train_loss("%s/train_loss.txt"%svdir, loss_list)
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
        if msg:
            U.message_board(Line+'\n', "%s/epochout.txt"%svdir)
            allepochlines.append(Line+"\n")

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

    # Run quick(ish) few shot downstream evaluation
    if config['downstream'] is not None:
        for dstask in config['downstream']:
            DS = allds[dstask](
                dsconfig, base_model=encoder, 
                svdir='%s/downstream/%s/'%(svdir, dstask)
            )
            sys.stdout.write("\r\033[KDownstream evlauation: %s\n"%(dstask))
            line, highline = DS.TrainEval()
            Line = "Downstream evlauation: %s; "%(dstask) + highline
            sys.stdout.write("\r\033[K%s\n"%Line)
            if msg:
                U.message_board("\n".join(line)+'\n', "%s/epochout.txt"%svdir)
                allepochlines.append(Line+"\n")
    if msg:
        # Append results to the .all files
        U.message_board("".join(allepochlines), "save/epochout.all")
    
    print()

if __name__ == '__main__':
    
    if config['seed'] is not None:
        np.random.seed(config['seed'])
        th.manual_seed(config['seed'])
    train(epochs=config['epochs'], runlen=100, svfreq=config['svfreq'])

