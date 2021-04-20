# sys
import os
import sys
import numpy as np
import random
import pickle

# time
import time
from datetime import timedelta

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# for hyperparameter
from collections import OrderedDict
from collections import namedtuple
from itertools import product

from data_load import Dataset, data_load
from transfer_model import transfer_model
from kl_anneal import kl_anneal_function

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def kl_loss(mean, logvar, batch_size):

    loss_kl = torch.sum(0.5 * (mean**2 + torch.exp(logvar)- 1. \
                - logvar)) /batch_size

    return loss_kl
    

def train(model, data_loader, optimizer, batch_size, predict_frames, params, 
            gradient_clip, max_norm, device, step, gamma):

    #loss_functions
    loss_fn1 = nn.L1Loss(reduction='sum')
    loss_fn2 = nn.MSELoss(reduction='sum')
    loss_train = []
    loss = 0
    labels = []
    embeddings = []

    model.train()

    for batch in data_loader['train']:
        # get data
        data, label = batch
        
        data = data.float().to(device)
        label = label.float().to(device)
        #print(label.shape)
        
        landmark_label = label[:,:,:19,:2]     #(N, 1, 19, 2)
        pose_label = label[:,:,:1,2:]         #(N, 1, 1, 3)
        # print(landmark_label.shape)
        yaw_label, pitch_label, roll_label = pose_label[:,:,:,0], \
                pose_label[:,:,:,1], pose_label[:,:,:,2]

        # forward
        pose_pred, landmark_pred, embedding = model(data, predict_frames=predict_frames)
        
        poselabel_for_vis = pose_label.reshape(label.size(0), 30*3)
        labels.append(poselabel_for_vis.cpu())
        embeddings.append(embedding.cpu().detach().numpy())
        
        yaw, pitch, roll = pose_pred[:,:,:,0], pose_pred[:,:,:,1], pose_pred[:,:,:,2]          
        
        # loss
        alpha, beta = 0.001,0.0015
        Pitch_loss = beta * loss_fn2(pitch, pitch_label)
        Yaw_loss = alpha * loss_fn2(yaw, yaw_label)
        Roll_loss = alpha * loss_fn2(roll, roll_label)

        loss_1 = loss_fn1(landmark_pred, landmark_label) /batch_size
        loss_2 = (Yaw_loss + Pitch_loss + Roll_loss) /batch_size

        # GMM loss
        loss_kl_y = model.gaussian_weights[0]*kl_loss(model.mean_y, model.logvar_y, batch_size)
        loss_kl_p = model.gaussian_weights[1]*kl_loss(model.mean_p, model.logvar_p, batch_size)
        loss_kl_r = model.gaussian_weights[2]*kl_loss(model.mean_r, model.logvar_r, batch_size)
        loss_kl = loss_kl_y + loss_kl_p + loss_kl_r
        
        #for kl_annealing
#         k = 0.05
#         x0 = 200
#         ann_function = 'linear'
#         gamma = kl_anneal_function(ann_function, step, k, x0)

        #for constant_anneal
        # gamma = 1e-4

        loss = loss_1 + loss_2 + gamma * loss_kl
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        if gradient_clip: nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step()
        
        step += 1

        # statistics
        loss_train.append(loss.data.item())

        del data
        del label
    
    labels = np.vstack(labels)
    embeddings = np.vstack(embeddings)
    
    train_loss = np.mean(loss_train)
    
    return train_loss, step, embeddings, labels

def evaluate(model, data_loader, batch_size, predict_frames, device):
    
    # valid time
    model.eval()

    loss_fn = nn.L1Loss(reduction='sum')
    loss_valid_mae = []
    loss_valid_kl = []
    loss_valid_yaw = []
    loss_valid_pitch = []
    loss_valid_roll = []

    
    for batch in data_loader['test']:
        data, label = batch
        # get data
        data = data.float().to(device)
        label = label.float().to(device)
        
#       landmark_label = label[:,:,:19,:2]     #(N, 1, 19, 2)
        pose_label = label[:,:,:1,2:]         #(N, 1, 1, 3)

        # inference
        with torch.no_grad():
            pose_pred, _, _ = model(data, predict_frames=predict_frames)
            valid_loss_yaw = loss_fn(pose_pred[:,:,:,0], pose_label[:,:,:,0]) /batch_size /predict_frames
            valid_loss_pitch = loss_fn(pose_pred[:,:,:,1], pose_label[:,:,:,1]) /batch_size /predict_frames
            valid_loss_roll = loss_fn(pose_pred[:,:,:,2], pose_label[:,:,:,2]) /batch_size /predict_frames
            
            valid_loss = loss_fn(pose_pred, pose_label) /batch_size /3 /predict_frames

            loss_kl_y = model.gaussian_weights[0]*kl_loss(model.mean_y, model.logvar_y, batch_size)
            loss_kl_p = model.gaussian_weights[1]*kl_loss(model.mean_p, model.logvar_p, batch_size)
            loss_kl_r = model.gaussian_weights[2]*kl_loss(model.mean_r, model.logvar_r, batch_size)
            loss_kl = loss_kl_y + loss_kl_p + loss_kl_r
            
        loss_valid_mae.append(valid_loss.item())
        loss_valid_kl.append(loss_kl.item())

        loss_valid_yaw.append(valid_loss_yaw.item())
        loss_valid_pitch.append(valid_loss_pitch.item())
        loss_valid_roll.append(valid_loss_roll.item())

        del data
        del label

    valid_loss_mae = np.mean(loss_valid_mae)
    valid_loss_kl = np.mean(loss_valid_kl)

    loss_yaw = np.mean(loss_valid_yaw)
    loss_pitch = np.mean(loss_valid_pitch)
    loss_roll = np.mean(loss_valid_roll)
    return valid_loss_mae, valid_loss_kl, loss_yaw, loss_pitch, loss_roll

def pipeline(data_path, model_type, embedd_dim, epoch_num, debug, batch_size, predict_frames, 
                learn_rate, gradient_clip, gamma, optim_type='SGD', dropout=0, YawPR_frame_interval=5, 
                gaze_frame_interval=2, device='cuda:0', pretrain=False, pipeline_mode='train'):

    SEED = 2020
    torch.manual_seed(SEED)
    
    hyperparams = OrderedDict(
        lr = [learn_rate],
        batch_size = [batch_size],
        device = [device],
        gamma = gamma
    )
    
    epoch_num = epoch_num
    require_improvement = epoch_num
    max_norm = 10

    train_data, train_label, valid_data, valid_label, Model, input_channels = data_load(model_type, \
                        data_path, YawPR_frame_interval, gaze_frame_interval)
    
    data_loader = dict()
    model = Model(in_channels=input_channels,
                edge_importance_weighting=True,
                predict_frames=predict_frames,
                embedd_dim=embedd_dim,
                device=device,
                dropout=dropout)
    
    if pretrain == 1:
        transfer_model('./checkpoints/CST_VGAE/pretrain_model_CST_VGAE_3.65_7.46.pkl', model, device)
    else:
        model.apply(weights_init)
          
    for run in RunBuilder.get_runs(hyperparams):

        model = model.to(run.device)
        
        #load data 
        data_loader['train'] = torch.utils.data.DataLoader(
                        dataset=Dataset(train_data,train_label,predict_frames=predict_frames, debug=debug),
                        batch_size=run.batch_size,
                        shuffle=True,
                        num_workers=2,
                        drop_last=True)
        data_loader['test'] = torch.utils.data.DataLoader(
                        dataset=Dataset(valid_data,valid_label,predict_frames=predict_frames, debug=debug),
                        batch_size=run.batch_size,
                        shuffle=False,
                        num_workers=2)
        
        params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if optim_type == 'SGD':
            optimizer = optim.SGD(
                        params,
                        lr=run.lr,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.0001)
            
        if optim_type == 'Adam':
            optimizer = optim.Adam(
                        params,
                        lr=run.lr,
                        weight_decay=0.0001)
        

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=4e-08)

        
        global total_iterations
        global best_validation_loss
        global last_improvement

        total_iterations = 0
        best_validation_loss = float('inf')
        last_improvement = 0
        
        # Start-time used for printing time-usage below.
        start_time = time.time()
        
        step = 0

        if pipeline_mode == 'train':

            print('pipeline mode:{}, batch_size:{}, learning_rate:{}, device:{}, kl_penalty:{}'.format(pipeline_mode, run.batch_size, run.lr, run.device, run.gamma))

            for epoch in range(epoch_num):
                
                total_iterations += 1
                train_loss, step, embeddings, labels = train(model, data_loader, optimizer, run.batch_size, predict_frames, params, gradient_clip, max_norm, run.device, step, run.gamma)
                
#                 print(embeddings.shape)
#                 print(labels.shape)

                valid_loss_mae, valid_loss_kl, loss_yaw, loss_pitch, loss_roll = evaluate(model, \
                        data_loader, run.batch_size, predict_frames, run.device)
                
                valid_loss = valid_loss_mae + valid_loss_kl
#                 scheduler.step(valid_loss)

                if valid_loss_mae < best_validation_loss:
                    best_validation_loss = valid_loss_mae
                    best_train_loss = train_loss
                    best_loss_yaw = loss_yaw
                    best_loss_pitch = loss_pitch
                    best_loss_roll = loss_roll
                    best_loss_total = valid_loss_mae
                    best_loss_kl = valid_loss_kl

                    last_improvement = total_iterations
                    improved_str = '*'
                    best_checkpoint = model.state_dict()
                    #save embeddings
#                     np.save('./checkpoints/CST_VGAE/CST_VGAE_embeddings_data.npy', embeddings)
#                     np.save('./checkpoints/CST_VGAE/CST_VGAE_labels_data.npy', labels)

                else:
                    improved_str = ''

                msg ='[epoch:{}] [train_loss:{}] [valid_total:{}] [valid_mae:{}] [valid_kl:{}] {}\n\
                        [yaw_loss:{}] [pitch_loss:{}] [roll_loss:{}]'
                if (epoch+1) %10 == 0:
                    print(msg.format(epoch+1, train_loss, valid_loss, valid_loss_mae, valid_loss_kl, \
                                    improved_str, loss_yaw, loss_pitch, loss_roll))

                if total_iterations - last_improvement > require_improvement:
                    print("No improvement found in a while, stopping optimization.")

                    # Break out from the for-loop.
                    break

        elif pipeline_mode == 'test':

            print('pipeline mode:{}, batch_size:{}, device: {}'.format(pipeline_mode, run.batch_size, run.device))

            best_validation_loss, best_loss_kl, best_loss_yaw, best_loss_pitch, best_loss_roll = evaluate(model, \
                    data_loader, run.batch_size, predict_frames, run.device)

            best_train_loss = 0.
            best_loss_total = best_validation_loss
                
             
            
        print('=============================================')
        # Ending time.
        end_time = time.time()
        
    
        # Difference between start and end-times.
        time_dif = end_time - start_time
        

        if best_validation_loss == float('inf'):
            best_validation_loss = 'nan'

        # save model 
        base_path = "./checkpoints"
        checkpoint_name = "{}_klPenalty{}_y{:.2f}_p{:.2f}_r{:.2f}_mae{:.2f}_kl{:.2f}.pth".format(model_type, run.gamma, best_loss_yaw, best_loss_pitch, best_loss_roll, best_validation_loss, best_loss_kl)
        path = os.path.join(base_path, model_type, checkpoint_name)
        torch.save(best_checkpoint, path)

        print(f'model_type:{model_type}, embedd_dim:{embedd_dim}, predict_frames:{predict_frames}\n')
        print(f'best_loss_valid:{best_validation_loss}, best_train_loss_mae:{best_train_loss}\n')
        print(f'best_loss_yaw:{best_loss_yaw}, best_loss_pitch:{best_loss_pitch}, best_loss_roll:{best_loss_roll}\n')
        print(f'best_loss_mae:{best_validation_loss}, best_loss_kl:{best_loss_kl}\n')
        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        print('==========================================================================================\n')
        
        # print(loss_dict)
    return best_loss_total, best_train_loss, time_dif, best_loss_yaw, best_loss_pitch, best_loss_roll, best_validation_loss, best_loss_kl

