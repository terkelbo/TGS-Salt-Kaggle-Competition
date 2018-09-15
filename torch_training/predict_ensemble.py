# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:54:52 2018

@author: terke
"""

#try mix of dice loss and bce 2*dice + 1*bces

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
from torch_models.albunet import get_model as get_model_34
from torch_models.unet_resnet import get_model as get_model_101
from metrics.metric_implementations import iou_metric_batch

import cv2

#training constants
parameter_path_list = ['CV5_resnet34_weighted_loss_no_drop','CV5_resnet101_weighted_loss_no_drop']
submission_name = 'CV5_resnet34-101_weighted_loss_no_drop_tta.csv'

models = [0,1,2,3,4]
weights = [0.75,0.25]

model_predictions = []

test_path = '../test'
#test files
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

for parameter_path in parameter_path_list:
    for j in models:
        #load best model
        if '34' in parameter_path:
            model = get_model_34(num_classes = 1, num_filters = 32, pretrained = True)
        else:
            model = get_model_101(encoder_depth = 101, num_classes = 1, num_filters=32, dropout_2d=0.2, pretrained=True, is_deconv=True)
        
        model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt'))

        #test predictions
        model.train(False)
        new_predictions = []
        with torch.no_grad():
            for image in tqdm(data.DataLoader(test_dataset, batch_size = 100)):
                image = image[0].type(torch.FloatTensor).cuda()
                image_flipped = torch.from_numpy(np.flip(image,axis=3).copy()).cuda()
                y_pred = model(image).cpu().data.numpy()
                y_pred_flipped = np.flip(model(image_flipped).cpu().data.numpy(),axis=3)
                new_predictions.append(y_pred/2 + y_pred_flipped/2)
        new_predictions_stacked = np.vstack(new_predictions)[:, 0, :, :]/len(models)

        if j == 0:
            all_predictions_stacked = new_predictions_stacked.copy()
        else:
            all_predictions_stacked = all_predictions_stacked + new_predictions_stacked

    #same size for all test images
    height, width = 101, 101

    #calculate padding
    x_min_pad, x_max_pad, y_min_pad, y_max_pad = shape_image(height, width)

    #Center cropping because resizing is done by reflection!!!!!
    all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]    

    model_predictions.append(all_predictions_stacked)

all_predictions_stacked = sum([preds*weight for preds, weight in zip(model_predictions,weights)])
        
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



