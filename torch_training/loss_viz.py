# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 10:57:48 2018

@author: terke
"""

import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

pickle_name = '../pkls/CV5_resnext50_weighted_loss_no_drop_low_pixels_two_stage_SE_stratified_on_plateau_adam_hyper_decoder_gated.pkl'

# Getting back the objects:
with open(pickle_name,'rb') as f:  # Python 3: open(..., 'rb')
    train_loss_save, val_loss_save, val_iou_save = pickle.load(f)
    
    
#compute the mean of each val batch of size 13
val_iou_df = pd.DataFrame.from_dict(val_iou_save)
epochs = 130
split =int(val_iou_df.shape[0]/epochs)
indicies = np.array([np.repeat(i,split) for i in range(epochs)]).reshape(-1)
val_iou_df['indicies'] = indicies
val_iou_df = val_iou_df.groupby('indicies').mean()

#
plt.figure()
val_iou_df.plot()
plt.ylim([0.55,0.85])
plt.show()

plt.figure()
val_iou_df.plot()
plt.xlabel('Epochs')
plt.ylabel('Validation Kaggle IOU')
plt.title('Unet with SE-ResNext-50 Encoder')
plt.savefig('Example_training_resnext50.png')
plt.show()

val_iou_df.max()