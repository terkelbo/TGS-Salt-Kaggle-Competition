# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:59:08 2018

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

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.unet11_model import get_model
from metrics.metric_implementations import iou_metric_batch

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

#define model
model = get_model(num_classes = 1, num_filters = 32, drop = 0.5, pretrained = False)

#training parameters
epoch = 50
learning_rate = 1e-3
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40],gamma=0.5,last_epoch=-1)

#early stopping params
patience = 5
best_loss = 1e15
best_iou = 0.0
i = 0

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
            val_iou.append(iou_metric_batch(mask.cpu().numpy(),y_pred.cpu().numpy())) 
            
    
    print("Epoch: %d, Train: %.3f, Val: %.3f, Val IOU: %.3f" % (e, np.mean(train_loss), np.mean(val_loss), np.mean(val_iou)))
    if np.mean(val_iou) > best_iou:
        torch.save(model.state_dict(), '../torch_parameters/best_model_unet11.pt') #save
        i = 0 #reset 
        best_iou = np.mean(val_iou) #reset
    elif i > patience:
        break
    
    scheduler.step()

#load best model
model = get_model(num_classes = 1, num_filters = 32, drop = 0.5, pretrained = False)
model.load_state_dict(torch.load('../torch_parameters/best_model_unet11.pt'))

#test predictions
model.train(False)
all_predictions = []
with torch.no_grad():
    for image in tqdm(data.DataLoader(test_dataset, batch_size = 80)):
        image = image[0].type(torch.FloatTensor).cuda()
        y_pred = model(image).cpu().data.numpy()
        all_predictions.append(y_pred)
all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]

#same size for all test images
height, width = 101, 101

#calculate padding
x_min_pad, x_max_pad, y_min_pad, y_max_pad = shape_image(height, width)

#Center cropping because resizing is done by reflection!!!!!
all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]

val_predictions = []
val_masks = []
with torch.no_grad():
    for image, mask in tqdm(data.DataLoader(dataset_val, batch_size = 30)):
        image = image.type(torch.FloatTensor).cuda()
        y_pred = model(image).cpu().data.numpy()
        val_predictions.append(y_pred)
        val_masks.append(mask.numpy().astype(int))
    
val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]

#Center cropping because resizing is done by reflection!!!!!
val_predictions_stacked = val_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
val_masks_stacked = val_masks_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]

metric_by_threshold = []
for threshold in np.linspace(0, 1, 11):
    val_binary_prediction = (val_predictions_stacked > threshold).astype(int)
    
    """
    iou_values = []
    for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
        iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
        iou_values.append(iou)
    iou_values = np.array(iou_values)
    
    accuracies = [
        np.mean(iou_values > iou_threshold)
        for iou_threshold in np.linspace(0.5, 0.95, 10)
    ]
    print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
    """
    metric_by_threshold.append((iou_metric_batch(val_masks_stacked, val_binary_prediction), threshold))
    
    
best_metric, best_threshold = max(metric_by_threshold)


threshold = best_threshold
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
submit.to_csv('../submissions/unet-vgg11-w-transposed.csv', index = False)



