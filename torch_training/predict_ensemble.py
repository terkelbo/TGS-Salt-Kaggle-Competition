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
from torch_models.albunet import get_model
from metrics.metric_implementations import iou_metric_batch

import cv2

#training constants
parameter_path = 'CV5_resnet34_weighted_loss'
submission_name = 'CV5_resnet34_weighted_loss.csv'


for j in range(5):
	#load best model
	model = get_model(num_classes = 1, num_filters = 32, pretrained = True)
	model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(j) + '.pt'))

	#test predictions
	model.train(False)
	new_predictions = []
	with torch.no_grad():
	    for image in tqdm(data.DataLoader(test_dataset, batch_size = 100)):
	        image = image[0].type(torch.FloatTensor).cuda()
	        y_pred = model(image).cpu().data.numpy()
	        new_predictions.append(y_pred)
	new_predictions_stacked = np.vstack(new_predictions)[:, 0, :, :]/fold.get_n_splits()

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



