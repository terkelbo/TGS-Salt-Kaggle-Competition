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
#from tqdm import tqdm_notebook as tqdm

import torch
from torch import nn
from torch.utils import data

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.binary_model import get_model
from metrics.metric_implementations import iou_metric_batch
from torch_loss.losses import FocalLoss, dice_loss, lovasz_hinge
from functions import get_mask_type

import cv2

#training constants
parameter_path = 'binary_model_resnet34'

if not os.path.isdir('../torch_parameters/' + parameter_path):
    os.mkdir('../torch_parameters/' + parameter_path)

#file name constants
train_path = '../train'
test_path = '../test'

#file list
depths_df = pd.read_csv('../train.csv')
file_list = list(depths_df['id'].values)

#redo mask list
mask_paths = [train_path + '/masks/' + str(file) + '.png' for file in file_list]
mask_class = [get_mask_type(np.transpose(cv2.imread(mask), (2, 0, 1))[0,:, :]/255) for mask in mask_paths]

#do the target of the model
target = [1 if mask_type == 0 else 0 for mask_type in mask_class]

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
dataset = TGSSaltDataset(train_path, file_list_train, augmentation = True, binary = True)
dataset_val = TGSSaltDataset(train_path, file_list_val, binary = True)
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True, binary = True)

#define model
model = get_model(num_classes = 1, num_filters = 32, pretrained = True)

#training parameters
epoch = 100
learning_rate = 1e-4
bceloss = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=0.5,mode='max')

#early stopping params
patience = 20
best_loss = 1e15
best_acc = 0


loss_fn = lambda pred, target: bceloss(pred,target)

#training procedure
for e in range(epoch):
    train_loss = []
    model.train(True)
    for image, target in tqdm(data.DataLoader(dataset, batch_size = 16, shuffle = True)):        
        image = image.type(torch.FloatTensor).cuda()
        y_pred = model(image).squeeze(3).squeeze(2)
        loss = loss_fn(y_pred, target.cuda())
        
        optimizer.zero_grad()              
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.item())
        
    val_loss = []
    val_acc = []
    model.train(False)
    with torch.no_grad():
        for image, target in data.DataLoader(dataset_val, batch_size = 16, shuffle = False):
            image, target = image.cuda(), target.cuda()
            y_pred = model(image).squeeze(3).squeeze(2)
    
            loss = loss_fn(y_pred, target)
            val_loss.append(loss.data.item())
            val_acc.append((( (y_pred > 0.5).long() == target.long()).sum().float()/len(target)).item()) 
            
    
    print("Epoch: %d, Train: %.3f, Val: %.3f, Val Acc: %.3f" % (e, np.mean(train_loss), np.mean(val_loss), np.mean(val_acc)))
    if np.mean(val_acc) > best_acc:
        torch.save(model.state_dict(), '../torch_parameters/' + parameter_path + '/model-' + str(0) + '.pt') #save
        i = 0 #reset 
        best_iou = np.mean(val_acc) #reset
#            elif i > patience:
#                break
    
    scheduler.step(np.mean(val_acc))

#load best model
model = get_model(num_classes = 1, num_filters = 32, pretrained = True)
model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(0) + '.pt'))


#test predictions
model.train(False)
predictions = []
with torch.no_grad():
    for image in tqdm(data.DataLoader(test_dataset, batch_size = 64)):
        image = image[0].type(torch.FloatTensor).cuda()
        y_pred = model(image).cpu().data.numpy()
        predictions.append(y_pred > 0.5)
predictions_stacked = np.vstack(predictions)[:, 0, :, :]

