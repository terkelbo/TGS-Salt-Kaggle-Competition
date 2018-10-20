# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:43:11 2018

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

from torch.nn import functional as F

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.albunet101_no_drop_SE import get_model as get_model_101
from torch_models.albunet152_no_drop_SE import get_model as get_model_152
from metrics.metric_implementations import iou_metric_batch
from torch_loss.losses import FocalLoss, dice_loss, lovasz_hinge
from functions import get_mask_type

import cv2

parameter_paths = ['CV5_resnet152_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam',
                   'CV5_resnet101_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam']

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
fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
folds = list(fold.split(file_list,mask_class))

train_idx = folds[0][0]
val_idx = folds[0][1]

#20 % train/val split
file_list_val = list(map(file_list.__getitem__,val_idx))
file_list_train = list(map(file_list.__getitem__,train_idx))

#define dataset iterators
dataset = TGSSaltDataset(train_path, file_list_train)
dataset_val = TGSSaltDataset(train_path, file_list_val)
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,  kernel_size = kernel_size, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)
    
class StackingFCN(nn.Module):
    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d

        self.conv = nn.Sequential(ConvBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)))

        self.final = nn.Sequential(nn.Conv2d(filter_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        x = F.dropout2d(self.conv(x), p=self.dropout_2d)
        return self.final(x)
    
def PredictFolds(parameter_path, model, dataset):
    predictions_list = []
    sigmoid = nn.Sigmoid()
    for model_name in os.listdir('../torch_parameters/' + parameter_path):
        model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/' + model_name))
        model.train(False)
        new_predictions = []
        with torch.no_grad():
            for image in tqdm(data.DataLoader(dataset, batch_size = 12, num_workers=1, shuffle = False)):
                image = image[0].type(torch.FloatTensor).cuda()
                image_flipped = torch.from_numpy(np.flip(image,axis=3).copy()).cuda()
                y_pred = sigmoid(model(image)).cpu().data.numpy()
                y_pred_flipped = np.flip(sigmoid(model(image_flipped)).cpu().data.numpy(),axis=3)
                new_predictions.append(y_pred/2 + y_pred_flipped/2)
                del image; torch.cuda.empty_cache();
        predictions_list.append(np.vstack(new_predictions))
    return predictions_list

model = get_model_152()
train_list_152 = PredictFolds(parameter_paths[0], model, dataset)
val_list_152 = PredictFolds(parameter_paths[0], model, dataset_val)

model = get_model_101()
train_list_101 = PredictFolds(parameter_paths[1], model, dataset)
val_list_101 = PredictFolds(parameter_paths[1], model, dataset_val)

list(data.DataLoader(dataset, batch_size = 64))

#training parameters
epoch = 100
learning_rate = 1e-4
bceloss = torch.nn.BCELoss()
focalloss = FocalLoss(gamma=0.25)
diceloss = dice_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 50, 60],gamma=0.5, last_epoch=-1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 1e-3,last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.5,mode='max')

loss_fn = lambda pred, target: lovasz_hinge(pred,target)
#early stopping params
patience = 20
best_loss = 1e15
best_iou = 0.0

sigmoid = nn.Sigmoid()

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