# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:57:17 2018

@author: terke
"""

import os
import sys
import glob
sys.path.append('../')
sys.path.append('./')

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from tqdm import tqdm

import torch
from torch.utils import data
from torchvision.transforms.functional import hflip, to_pil_image, to_tensor

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.albunet_no_drop import get_model
from metrics.metric_implementations import iou_metric_batch
from torch_loss.losses import FocalLoss, dice_loss

import cv2

#training constants
parameter_path = 'CV5_resnet34_weighted_loss_no_drop'
submission_name = 'CV5_resnet34_weighted_loss_no_drop.csv'

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
num_pixels = [[mask,(cv2.imread(mask) < 128).sum()] for mask in mask_paths]
idx = np.argsort(np.array(num_pixels)[:,1].astype(int))
num_pixels_sorted = np.array(num_pixels)[idx]
num_pixels_sorted[:20]
cutoff = 40
removed_images = [path.split('/')[-1][:-4] for path, nb_pixels in num_pixels_sorted if int(nb_pixels) < cutoff]

#redo file list
#file_list = [file for file in file_list if file not in removed_images]

#test files
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 

#five fold generator
fold = KFold(n_splits = 5, shuffle = True, random_state = 42)

for j, idx in enumerate(fold.split(file_list)):
    j = 2
    train_idx = idx[0]
    val_idx = idx[1]

    #20 % train/val split
    file_list_val = list(map(file_list.__getitem__,val_idx))
    file_list_train = list(map(file_list.__getitem__,train_idx))

    #define dataset iterators
    dataset = TGSSaltDataset(train_path, file_list_train, augmentation = True)
    dataset_val = TGSSaltDataset(train_path, file_list_val)
    test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

    if '34' in parameter_path:
        model = get_model(num_classes = 1, num_filters = 32, pretrained = True)
    else:
        model = get_model(encoder_depth = 101, num_classes = 1, num_filters=32, dropout_2d=0.2, pretrained=True, is_deconv=True)
    
    model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt'))
    model.train(False)


    #training parameters
    epoch = 100
    learning_rate = 1e-3
    bceloss = torch.nn.BCELoss()
    diceloss = dice_loss
    loss_fn = lambda pred, target: 2*dice_loss(pred, target) + bceloss(pred,target)

    #early stopping params
    patience = 20
    best_loss = 1e15
    best_iou = 0.0
    i = 0
    
    val_loss = []
    val_iou = []
    with torch.no_grad():
        for image, mask in data.DataLoader(dataset_val, batch_size = 128, shuffle = False):
            image = image.cuda()
            y_pred = model(image)
    
            loss = loss_fn(y_pred, mask.cuda())
            val_loss.append(loss.data.item())
            val_iou.append(iou_metric_batch(mask.cpu().numpy(),y_pred.cpu().numpy())) 
            
    print("Val: %.3f, Val IOU: %.3f" % (np.mean(val_loss), np.mean(val_iou)))
