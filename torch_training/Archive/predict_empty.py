# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:49:53 2018

@author: terke
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

import torch
from torch import nn
from torch.utils import data

from tqdm import tqdm

from torch_dataset.dataset_prep import TGSSaltDataset, shape_image
from torch_models.binary_model import get_model

sys.path.append('./')
sys.path.append('..')

submission_name = 'CV5_resnet101-152_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam-tta.csv'
parameter_path = 'binary_model_resnet34'

submission = pd.read_csv('../submissions/' + submission_name)

submission.describe()

#load best model
model = get_model(num_classes = 1, num_filters = 32, pretrained = True)
model.load_state_dict(torch.load('../torch_parameters/' + parameter_path + '/model-' + str(0) + '.pt'))

#test files
test_path = '../test'
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('\\')[-1][:-4] for f in test_file_list] 
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

#test predictions
model.train(False)
predictions = []
with torch.no_grad():
    for image in tqdm(data.DataLoader(test_dataset, batch_size = 64)):
        image = image[0].type(torch.FloatTensor).cuda()
        y_pred = model(image).cpu().data.numpy()
        predictions.append(y_pred > 0.5)
predictions_stacked = np.vstack(predictions)[:, 0, :, :]

submit = pd.DataFrame([test_file_list, list(predictions_stacked.astype(int).reshape(-1))]).T
submit.columns = ['id', 'rle_mask']

combined = pd.merge(submission,submit,on='id')
combined['rle_mask'] = combined['rle_mask_x'].fillna('')
combined.loc[combined['rle_mask_y']==1,'rle_mask'] = ''

submision_out = combined[['id','rle_mask']]


submision_out.to_csv('../submissions/' + 'binary-empty-' + submission_name, index = False)

submit.loc[submit['rle_mask'] == 1,'rle_mask'] = '1 1'
submit.loc[submit['rle_mask'] == 0,'rle_mask'] = ''
submit.to_csv('../submissions/' + 'test_classifier.csv', index = False)

submission = submission.fillna('')
submission.loc[submission['rle_mask'] != '','rle_mask_new'] = ''
submission.loc[submission['rle_mask'] == '','rle_mask_new'] = '1 1'
submission.drop('rle_mask',axis=1,inplace=True)
submission.columns = ['id','rle_mask']
submission.to_csv('../submissions/' + 'test_unet_classifier.csv', index = False)
