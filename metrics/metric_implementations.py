# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 22:24:19 2018

@author: s144299
"""

import numpy as np
from sklearn.metrics import jaccard_similarity_score

def kaggle_iou(predicted_mask_list, true_mask_list):
    """ Implementation of the Kaggle metric.
    
        Defines the "mean average precision at different intersection over union (IoU) thresholds"
        
        Input is given two lists of numpy arrays
    """
    
    score = []
    for pred_mask, true_mask in zip(predicted_mask_list, true_mask_list):
        iou = jaccard_similarity_score(true_mask.flatten(), pred_mask.flatten())
        score.append(np.mean([1 if iou > threshold else 0 for threshold in np.linspace(0.5,1,11)]))
    return np.mean(score)