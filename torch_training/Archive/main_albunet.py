# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:54:52 2018

@author: terke
"""

import os
import sys
import glob
sys.path.append('../')
sys.path.append('../torch_models/')
sys.path.append('./')

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.albunet152_SE_hyper import get_model
from metrics.metric_implementations import iou_metric_batch
from torch_loss.losses import FocalLoss, dice_loss, lovasz_hinge
from functions import get_mask_type

import cv2
import pickle

#training constants
parameter_path = 'CV5_resnet152_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam_hyper_decoder'
submission_name = 'CV5_resnet152_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_hyper_decoder-tta.csv'

if not os.path.isdir('../torch_parameters/' + parameter_path):
    os.mkdir('../torch_parameters/' + parameter_path)

#file name constants
train_path = '../train'
test_path = '../test'

#file list
depths_df = pd.read_csv('../train.csv')
file_list = list(depths_df['id'].values)

#remove images from train with only few pixels in mask
mask_paths = [train_path + '/masks/' + str(file) + '.png' for file in file_list]
num_pixels = [[mask,(cv2.imread(mask)[:,:,0:1] > 128).sum()] for mask in mask_paths]
idx = np.argsort(np.array(num_pixels)[:,1].astype(int))
cutoff = 10
removed_images = [path.split('/')[-1][:-4] for path, nb_pixels in num_pixels if int(nb_pixels) < cutoff and int(nb_pixels) != 0]

#redo file list
file_list = [file for file in file_list if file not in removed_images]

#redo mask list
mask_paths = [train_path + '/masks/' + str(file) + '.png' for file in file_list]
mask_class = [get_mask_type(np.transpose(cv2.imread(mask), (2, 0, 1))[0,:, :]/255) for mask in mask_paths]

#test files
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 

#five fold generator
fold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

train_loss_save = {}
val_loss_save = {}
val_iou_save = {}

for j, idx in enumerate(fold.split(file_list,mask_class)):
    train_loss_save['fold_' + str(j)] = []
    val_loss_save['fold_' + str(j)] = []
    val_iou_save['fold_' + str(j)] = []
    
    
    train_idx = idx[0]
    val_idx = idx[1]

    #20 % train/val split
    file_list_val = list(map(file_list.__getitem__,val_idx))
    file_list_train = list(map(file_list.__getitem__,train_idx))

    #define dataset iterators
    dataset = TGSSaltDataset(train_path, file_list_train, augmentation = True)
    dataset_val = TGSSaltDataset(train_path, file_list_val)
    test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

    #define model
    model = get_model(num_classes = 1, num_filters = 32, pretrained = True)

    #training parameters
    epoch = 100
    learning_rate = 1e-4
    bceloss = torch.nn.BCELoss()
    focalloss = FocalLoss(gamma=0.25)
    diceloss = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 50, 60],gamma=0.5, last_epoch=-1)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 1e-3,last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.5,mode='max')

    #early stopping params
    patience = 20
    best_loss = 1e15
    best_iou = 0.0
    
    
    semi_threshold = epoch # = epoch => dont use pseudolabelling 
    
    sigmoid = nn.Sigmoid()

    for opt_round in range(2):
        i = 0
        if opt_round == 0:
            loss_fn = lambda pred, target: 5*diceloss(sigmoid(pred),target) + bceloss(sigmoid(pred),target)
        else:
            patience = 30
            loss_fn = lambda pred, target: lovasz_hinge(pred,target)
            learning_rate = 1e-5
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [33,66],gamma=0.1,last_epoch=-1)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 50, 60],gamma=0.5, last_epoch=-1)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 1e-4,last_epoch=-1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.5,mode='max')
      
        print('Starting training loop')
        #training procedure
        for e in range(epoch):
            train_loss = []
            model.train(True)
            for image, mask in tqdm(data.DataLoader(dataset, batch_size = 16, shuffle = True)):        
                image = image.type(torch.FloatTensor).cuda()
                y_pred = model(image)
                loss = loss_fn(y_pred, mask.cuda())
                
                optimizer.zero_grad()              
                loss.backward()
                optimizer.step()
                #scheduler.step()
                train_loss.append(loss.data.item())
                
            val_loss = []
            val_iou = []
            model.train(False)
            i += 1 #increment training step
            with torch.no_grad():
                for image, mask in data.DataLoader(dataset_val, batch_size = 64, shuffle = False):
                    image = image.cuda()
                    y_pred = model(image)
            
                    loss = loss_fn(y_pred, mask.cuda())
                    val_loss.append(loss.data.item())
                    val_iou.append(iou_metric_batch(mask.cpu().numpy(),sigmoid(y_pred).cpu().numpy())) 
                    
            
            print("Epoch: %d, Train: %.3f, Val: %.3f, Val IOU: %.3f" % (e, np.mean(train_loss), np.mean(val_loss), np.mean(val_iou)))
            if np.mean(val_iou) > best_iou:
                torch.save(model.state_dict(), '../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt') #save
                i = 0 #reset 
                best_iou = np.mean(val_iou) #reset
#            elif i > patience:
#                break
            
            scheduler.step(np.mean(val_iou))
            
            #save scores in dictionary
            train_loss_save['fold_' + str(j)].extend(train_loss)
            val_loss_save['fold_' + str(j)].extend(val_loss)
            val_iou_save['fold_' + str(j)].extend(val_iou)

    #load best model
    model = get_model(num_classes = 1, num_filters = 32, pretrained = True)
    model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt'))

    #test predictions
    model.train(False)
    new_predictions = []
    with torch.no_grad():
        for image in tqdm(data.DataLoader(test_dataset, batch_size = 64)):
            image = image[0].type(torch.FloatTensor).cuda()
            image_flipped = torch.from_numpy(np.flip(image,axis=3).copy()).cuda()
            y_pred = sigmoid(model(image)).cpu().data.numpy()
            y_pred_flipped = np.flip(sigmoid(model(image_flipped)).cpu().data.numpy(),axis=3)
            new_predictions.append(y_pred/2 + y_pred_flipped/2)
    new_predictions_stacked = np.vstack(new_predictions)[:, 0, :, :]/fold.get_n_splits()

    if j == 0:
        all_predictions_stacked = new_predictions_stacked
    else:
        all_predictions_stacked = all_predictions_stacked + new_predictions_stacked

#same size for all test images
height, width = 101, 101

#calculate padding
x_min_pad, x_max_pad, y_min_pad, y_max_pad = shape_image(height, width)

#Center cropping because resizing is done by reflection!!!!!
all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]    
    
threshold = 0.5
binary_prediction = (all_predictions_stacked > threshold).astype(int)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

all_masks = []
for p_mask in list(binary_prediction):
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))

submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('../submissions/' + submission_name, index = False)

#dump pickle files
with open('../pkls/' + parameter_path + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_loss_save, val_loss_save, val_iou_save], f)



