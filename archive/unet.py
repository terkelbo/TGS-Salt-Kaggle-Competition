# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 13:09:00 2018

@author: s144299
"""

import os
import sys
from itertools import chain

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import cv2

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gc

#file name constants
path_train = './train/'
path_test = './test/'

#other constants
im_width = 128
im_height = 128
border = 5
im_chan = 3 
n_features = 1 # Number of extra features, like depth

#depths file, includes both train and test set
depths = pd.read_csv('./depths.csv', index_col='id')
depths.head()

#train and test file names (id's)
train_ids = next(os.walk(path_train+'images'))[2]
test_ids = next(os.walk(path_test+'images'))[2]

#check first image
sample_img = imread(path_train + 'images/' + train_ids[1])
#imshow(sample_img)

#check mask
sample_mask = imread(path_train + 'masks/' + train_ids[1])
#imshow(sample_mask)

#normalize the depth
depths = (depths - depths.mean(axis=0))/depths.std(axis=0)

#get train images
# Get and resize train images and masks
#"""
X = np.zeros((len(train_ids), im_chan, im_height, im_width), dtype=np.float32)
y = np.zeros((len(train_ids), 1, im_height, im_width), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
    # Depth
    X_feat[n] = depths.loc[id_.replace('.png', ''), 'z']
    
    # Load image
    img = imread(path_train + '/images/' + id_)
    x_img = resize(img, (im_chan, im_height, im_width), mode='constant', preserve_range=True)
    
    # Load mask
    mask = imread(path_train + '/masks/' + id_)
    mask = resize(mask, (1, im_height, im_width), mode='constant', preserve_range=True)

    # Save images
    X[n] = x_img.squeeze() / 255
    y[n] = mask / 255
    
# Saving the objects:
with open('train_img.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([X,y,X_feat], f)
#"""
    
# Getting back the objects:
with open('train_img.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X, y, X_feat = pickle.load(f)
    
#"""
X_test = np.zeros((len(test_ids), im_chan, im_height, im_width), dtype=np.float32)
X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
sizes_test = []
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    
    # Depth
    X_feat_test[n] = depths.loc[id_.replace('.png', ''), 'z']
    
    # Load image
    img = imread(path_test + '/images/' + id_)
    
    sizes_test.append([img.shape[0], img.shape[1]])
    x_img = resize(img, (im_chan, im_height, im_width), mode='constant', preserve_range=True)
    
    # Save images
    X_test[n] = x_img.squeeze() / 255

# Saving the objects:
with open('test_img.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([X_test,X_feat_test, sizes_test], f)
#"""

# Getting back the objects:
with open('test_img.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X_test, X_feat_test, sizes_test = pickle.load(f)

#unet model
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_conv, self).__init__()

        if bilinear: #either use bilinear 
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else: #else ise transposed conv
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.double_conv1 = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        """ x1 is the current input and x2 is the copied from downsampling """
        x1 = self.up(x1)
        #calculate difference for padding
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        #pad input from downsampling
        x2 = F.pad(input = x2, pad = (diffX // 2, int(diffX / 2),
                                      diffY // 2, int(diffY / 2)),
                   mode = 'constant',
                   value = 0
            )
        #concatenate along channel axis
        x1 = torch.cat([x2, x1], dim=1)
        #double convolution
        x1 = self.double_conv1(x1)
        return x1

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        
        self.in_conv = double_conv(n_channels, 32)
        
        self.double_conv1 = double_conv(32, 64)
        self.double_conv2 = double_conv(64, 128)
        self.double_conv3 = double_conv(128, 256)
        
        self.up_conv1 = up_conv(384, 128)
        self.up_conv2 = up_conv(192, 64)
        self.up_conv3 = up_conv(96, 32)
        
        self.out_conv =  nn.Conv2d(32, n_classes-1, kernel_size = (1,1)) #ignore background
    
        self.max_pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))

        self.sigmoid = nn.Sigmoid()   
        
    def forward(self, x):
        x1 = self.in_conv(x)
        
        x1 = self.max_pool(x1)
        x2 = self.double_conv1(x1)
        
        x2 = self.max_pool(x2)
        x3 = self.double_conv2(x2)
        
        x3 = self.max_pool(x3)
        x4 = self.double_conv3(x3)
        
        #upsampling
        x = self.up_conv1(x4, x3)
        x = self.up_conv2(x, x2)
        x = self.up_conv3(x, x1)
        
        #out
        x = self.out_conv(x)
        
        return self.sigmoid(x)
        

#train/val split
X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size = 0.15,
                                                  random_state = 42)

#Create tensor, 0.4.0 variables are not needed
X_train = Tensor(X_train)
y_train = Tensor(y_train)

X_val = Tensor(X_val)
y_val = Tensor(y_val)

X_test = Tensor(X_test)

if torch.cuda.is_available():
    X_train, y_train = X_train.cuda(), y_train.cuda()
    X_val, y_val = X_val.cuda(), y_val.cuda()

#dataloaders
mini_batch_size = 32
X_train_loader = DataLoader(X_train,batch_size=mini_batch_size)
y_train_loader = DataLoader(y_train,batch_size=mini_batch_size)

X_val_loader = DataLoader(X_val,batch_size=256)
y_val_loader = DataLoader(y_val,batch_size=256)

#X_test_loader = DataLoader(X_test,batch_size=mini_batch_size)

del X_train, y_train#, X_val, y_val
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

#Training 10 times and getting average to avoid random errors
net_run_list = list()
for net_runs in range(1):
    net = UNet(n_channels = 3, n_classes = 2)

    #return_n_params(net)
    loss_f = nn.BCELoss()

    if torch.cuda.is_available():
        net.cuda()
        loss_f.cuda() #loss has to be moved to cuda for some special types of loss
        
    #Using ADAM optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    steps = 2

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 7, 9],gamma=0.1,last_epoch=-1)

    loss_save_train = list()
    loss_save_val = list()
    
    #Simple Training loop
    for i in tqdm(range(steps)):
        for X_train_batch, y_train_batch in zip(X_train_loader, y_train_loader):
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            net.train(True)
            out = net(X_train_batch)
            l = loss_f(out, y_train_batch)
            
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        if np.mod(i,2) == 0:    
            with torch.no_grad():
                t_out = torch.Tensor(0,1,128,128)
                if torch.cuda.is_available(): t_out = t_out.cuda()
                for X_val_batch, y_val_batch in zip(X_val_loader, y_val_loader):
                     t_out = torch.cat([t_out,net(X_val_batch)],0)
                tl = loss_f(t_out,y_val)
                print('epoch {0}: trainloss: {1}, valloss:{2}'.format(i,l.data.item(),tl.data.item()))
            
        scheduler.step()
        
    net.train(False)
    
    del X_train_loader, y_train_loader, X_val_loader, y_val_loader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
         X_test = X_test.cuda()
    
    X_test_loader = DataLoader(X_test,batch_size=256)

    del X_test
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    with torch.no_grad():    
        out_test = torch.Tensor(0, 1, 128, 128)
        for X_test_batch in tqdm(X_test_loader):
             pred = net(X_test_batch)
             out_test = torch.cat([out_test, pred.cpu()], 0)
             if torch.cuda.is_available(): torch.cuda.empty_cache()

    d = {'loss_save_train':loss_save_train,
         'loss_save_val':loss_save_val,
         'predictions':out_test}
    
    net_run_list.append(d)

#resample test mask
predictions = d['predictions'].squeeze(1).cpu().numpy()
print(predictions[0])
preds_test = []
for i in range(predictions.shape[0]):
    preds_test.append(resize(predictions[i], 
                            (sizes_test[i][0], sizes_test[i][1]), 
                            mode='constant', preserve_range=True))

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
    


    
pred_dict = {id_[:-4]:RLenc((preds_test[i] > 0.5).astype(int)) for i,id_ in tqdm(enumerate(test_ids[:5]))}

# Saving the objects:
with open('pred_dict.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([pred_dict], f)

print(pred_dict)

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('unet-initial.csv')
