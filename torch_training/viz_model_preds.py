# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:57:45 2018

@author: terke
"""

import os
import sys
import glob
sys.path.append('../')
sys.path.append('./')

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils import data
from torchvision.transforms.functional import hflip, to_pil_image, to_tensor

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.albunet import get_model
from metrics.metric_implementations import iou_metric_batch

import matplotlib.pyplot as plt

import cv2

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

#10 % train/val split
file_list_val = file_list[::10]  #every 10th image into validation
file_list_train = [f for f in file_list if f not in file_list_val]

#test files
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 

#define dataset iterators
dataset = TGSSaltDataset(train_path, file_list_train, augmentation = True)
dataset_val = TGSSaltDataset(train_path, file_list_val)
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

#load best model
model = get_model(num_classes = 1, num_filters = 32, pretrained = True)
model.load_state_dict(torch.load('../torch_parameters/best_model_albunet_drop_pretrained.pt'))

model.train(False)
image = test_dataset.__getitem__(555)
with torch.no_grad():
    y_pred = model(image[0].unsqueeze(0).cuda()).cpu().data.numpy()
    y_pred_flipped = to_tensor(hflip(to_pil_image(model(to_tensor(hflip(to_pil_image(image[0]))).unsqueeze(0).cuda())[0].cpu())))
plt.figure()
plt.imshow(y_pred[0,0,:,:] > 0.5,cmap='gray')
plt.figure()
plt.imshow(y_pred_flipped[0,:,:] > 0.5,cmap='gray')
plt.figure()
plt.imshow(((y_pred[0,0,:,:] + y_pred_flipped[0,:,:])/2>0.5).numpy(),cmap='gray')

