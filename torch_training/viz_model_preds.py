# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:57:45 2018

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

import torch
from torch import nn

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.resnext50_hyper_gated import get_model as get_model_50
from torch_models.resnext101_hyper_gated import get_model as get_model_101
from functions import get_mask_type

import matplotlib.pyplot as plt
import cv2

from mpl_toolkits.axes_grid1 import ImageGrid



#training constants
parameter_path_50 = 'CV5_resnext50_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam_hyper_decoder_gated'
parameter_path_101 = 'CV5_resnext101_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam_hyper_decoder_gated'

#file name constants
train_path = '../train'
test_path = '../test'

#file list
depths_df = pd.read_csv('../train.csv')
file_list = list(depths_df['id'].values)

#redo mask list
mask_paths = [train_path + '/masks/' + str(file) + '.png' for file in file_list]
mask_class = [get_mask_type(np.transpose(cv2.imread(mask), (2, 0, 1))[0,:, :]/255) for mask in mask_paths]

#test files
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 

#five fold generator
fold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

idx = list(fold.split(file_list,mask_class))

#fold to test
fold_to_test = 0

#grab train and val data from fold to test
train_idx = idx[fold_to_test][0]
val_idx = idx[fold_to_test][1]

#20 % train/val split
file_list_val = list(map(file_list.__getitem__,val_idx))
file_list_train = list(map(file_list.__getitem__,train_idx))

#redo mask list
mask_paths = [train_path + '/masks/' + str(file) + '.png' for file in file_list_train]
mask_class = [get_mask_type(np.transpose(cv2.imread(mask), (2, 0, 1))[0,:, :]/255) for mask in mask_paths]

#define dataset iterators
dataset = TGSSaltDataset(train_path, file_list_train, augmentation = True, classes = mask_class)
dataset_val = TGSSaltDataset(train_path, file_list_val)

#define resnext50
model_50 = get_model_50(num_classes = 1, num_filters = 32, pretrained = True)
model_50.load_state_dict(torch.load('../torch_parameters/' + parameter_path_50 + '/model-' + str(fold_to_test) + '.pt'))

#define resnext101
model_101 = get_model_101(num_classes = 1, num_filters = 32, pretrained = True)
model_101.load_state_dict(torch.load('../torch_parameters/' + parameter_path_101 + '/model-' + str(fold_to_test) + '.pt'))

sigmoid = nn.Sigmoid()

#test predictions
model_50.train(False)
model_101.train(False)

#init the collage
im = np.arange(100)
im.shape = 10, 10
images = [im for i in range(20)]

fig = plt.figure(1, (20., 5.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 10),
                 axes_pad=0.5
)

#grab 10 random images
val_idxs = np.random.choice(range(0,len(val_idx)),10)

for i in range(len(val_idxs)):
    val_image_idx = val_idxs[i]
    val_image_id = file_list_val[val_image_idx]
    image, mask = dataset_val.__getitem__(val_image_idx)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        image = image.type(torch.FloatTensor).cuda()
        image_flipped = torch.from_numpy(np.flip(image,axis=3).copy()).cuda()
        y_pred_50 = sigmoid(model_50(image)[0]).cpu().data.numpy()
        y_pred_flipped_50 = np.flip(sigmoid(model_50(image_flipped)[0]).cpu().data.numpy(),axis=3)
        
        y_pred_101 = sigmoid(model_101(image)[0]).cpu().data.numpy()
        y_pred_flipped_101 = np.flip(sigmoid(model_101(image_flipped)[0]).cpu().data.numpy(),axis=3)
        
        #final pred
        y_pred = ((y_pred_50 + y_pred_flipped_50 + y_pred_101 + y_pred_flipped_101)/4 > 0.5).astype(int)
    
    #recrop image, mask and pred
    height, width = 101, 101
    x_min_pad, x_max_pad, y_min_pad, y_max_pad = shape_image(height, width)
    
    image = image[0,:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]    
    mask = mask[0, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]    
    y_pred = y_pred[0, 0, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]    
    
    grid[i].imshow(np.transpose(image, (1, 2, 0)))
    grid[i].axis('off')
    grid[i].set_title(val_image_id)
    grid[i+10].imshow(mask, cmap='gray', interpolation='none')
    grid[i+10].imshow(y_pred, cmap=plt.get_cmap('Greys_r'), alpha=0.5, interpolation='none')  
    grid[i+10].axis('off')

plt.savefig('collage_fold_0.png')
plt.show()