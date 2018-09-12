# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:05:33 2018

@author: TerkelBo
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

from torch_dataset.dataset_prep import TGSSaltDataset
from torch_models.vgg11_unet_transposed import get_model, shape_image


#file name constants
train_path = '../train'
test_path = '../test'

#file list
depths_df = pd.read_csv('../train.csv')
file_list = list(depths_df['id'].values)

#10 % train/val split
file_list_val = file_list[::10]  #every 10th image into validation
file_list_train = [f for f in file_list if f not in file_list_val]

#test foæes
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]

#define dataset iterators
dataset = TGSSaltDataset(train_path, file_list_train, augmentation = True)
dataset_val = TGSSaltDataset(train_path, file_list_val)
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

#define model
model = get_model(3,2)

#training parameters
epoch = 50
learning_rate = 1e-3
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40],gamma=0.5,last_epoch=-1)

#training procedure
for e in range(epoch):
    train_loss = []
    model.train(True)
    for image, mask in tqdm(data.DataLoader(dataset, batch_size = 32, shuffle = True)):        
        image = image.type(torch.FloatTensor).cuda()
        y_pred = model(image)
        loss = loss_fn(y_pred, mask.cuda())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss.append(loss.data.item())
        
    val_loss = []
    model.train(False)
    with torch.no_grad():
        for image, mask in data.DataLoader(dataset_val, batch_size = 64, shuffle = False):
            image = image.cuda()
            y_pred = model(image)
    
            loss = loss_fn(y_pred, mask.cuda())
            val_loss.append(loss.data.item())
    
        print("Epoch: %d, Train: %.3f, Val: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)))
    
    scheduler.step()

#test predictions
model.train(False)
all_predictions = []
with torch.no_grad():
    for image in tqdm(data.DataLoader(test_dataset, batch_size = 100)):
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


from sklearn.metrics import jaccard_similarity_score

metric_by_threshold = []
for threshold in np.linspace(0, 1, 11):
    val_binary_prediction = (val_predictions_stacked > threshold).astype(int)
    
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
    metric_by_threshold.append((np.mean(accuracies), threshold))
    
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
submit.to_csv('../submissions/unet-vgg11.csv', index = False)



