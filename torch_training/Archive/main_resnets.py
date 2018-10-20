# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:23:34 2018

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
from torch_models.unet_resnet import get_model
from metrics.metric_implementations import iou_metric_batch
from torch_loss.losses import FocalLoss, dice_loss

import cv2

#training constants
parameter_path = 'CV5_resnet101_weighted_loss_no_drop'
submission_name = 'CV5_resnet101_weighted_loss_no_drop.csv'

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
num_pixels = [[mask,(cv2.imread(mask) > 128).sum()] for mask in mask_paths]
idx = np.argsort(np.array(num_pixels)[:,1].astype(int))
num_pixels_sorted = np.array(num_pixels)[idx]
num_pixels_sorted[:20]
cutoff = 40
removed_images = [path.split('/')[-1][:-4] for path, nb_pixels in num_pixels_sorted if int(nb_pixels) < cutoff]

#redo file list
file_list = [file for file in file_list if file not in removed_images]

#test files
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 

#five fold generator
fold = KFold(n_splits = 5, shuffle = True, random_state = 42)

for j, idx in enumerate(fold.split(file_list)):
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
    model = get_model(encoder_depth = 101, num_classes = 1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True)

    #training parameters
    epoch = 100
    learning_rate = 1e-3
    bceloss = torch.nn.BCELoss()
    diceloss = dice_loss
    loss_fn = lambda pred, target: 2*dice_loss(pred, target) + bceloss(pred,target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [33,66],gamma=0.1,last_epoch=-1)

    #early stopping params
    patience = 20
    best_loss = 1e15
    best_iou = 0.0
    i = 0

    #training procedure
    for e in range(epoch):
        train_loss = []
        model.train(True)
        for image, mask in tqdm(data.DataLoader(dataset, batch_size = 64, shuffle = True)):        
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
            for image, mask in data.DataLoader(dataset_val, batch_size = 128, shuffle = False):
                image = image.cuda()
                y_pred = model(image)
        
                loss = loss_fn(y_pred, mask.cuda())
                val_loss.append(loss.data.item())
                val_iou.append(iou_metric_batch(mask.cpu().numpy(),y_pred.cpu().numpy())) 
                
        
        print("Epoch: %d, Train: %.3f, Val: %.3f, Val IOU: %.3f" % (e, np.mean(train_loss), np.mean(val_loss), np.mean(val_iou)))
        if np.mean(val_iou) > best_iou:
            torch.save(model.state_dict(), '../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt') #save
            i = 0 #reset 
            best_iou = np.mean(val_iou) #reset
        elif i > patience:
            break
        
        scheduler.step()


    #load best model
    model = get_model(encoder_depth = 101, num_classes = 1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True)
    model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt'))

    #test predictions
    model.train(False)
    new_predictions = []
    with torch.no_grad():
        for image in tqdm(data.DataLoader(test_dataset, batch_size = 128)):
            image = image[0].type(torch.FloatTensor).cuda()
            y_pred = model(image).cpu().data.numpy()
            new_predictions.append(y_pred)
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
