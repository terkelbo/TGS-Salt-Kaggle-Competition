# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:28:10 2018

@author: s144299
"""

import os

import torch
from torch.utils import data
from torchvision import transforms

import numpy as np

import cv2

from albumentations import (
        ShiftScaleRotate,
        RandomSizedCrop,
        ElasticTransform,
        GridDistortion,
        RandomBrightness,
        RandomGamma,
        Blur,
        GaussNoise,
        HorizontalFlip,
        Compose,
        OneOf, 
        PadIfNeeded
)

def shape_image(height, width):
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
    
    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
        
    return x_min_pad, x_max_pad, y_min_pad, y_max_pad 
        
    
def load_image(path, mask = False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR to RGB
    
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return img
    else:
        img = img / 255.0
        return img
    
    
    
class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test = False, augmentation = False, binary = False, classes = None):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        self.augmentation = augmentation
        self.binary = binary
        self.classes = classes
        
        #data augmentation
        self.shiftscalerotate = ShiftScaleRotate(p = 0.25)
        self.randomsizecrop = RandomSizedCrop(min_max_height=(50,101), height = 128, width = 128,p=0.25)
        self.elastic = ElasticTransform(p=0.25, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        self.griddistortion = GridDistortion(p = 0.25)
        self.randombrighness = RandomBrightness(p=0.25)
        self.randomgamma = RandomGamma(p = 0.25)
        self.brightness = RandomBrightness(p = 0.25)
        self.blur = Blur(p = 0.25, blur_limit = 5)
        
        self.pad = PadIfNeeded(p = 1, min_height = 128, min_width = 128)
        
        self.compose = Compose([HorizontalFlip(p=0.5),
                    OneOf([self.shiftscalerotate, self.randomsizecrop, self.elastic], p = 1), 
                    OneOf([self.griddistortion, self.randombrighness, self.randomgamma, self.brightness,self.blur], p = 1)], 
                    p = 0.8)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        image = load_image(image_path)
        
        if not self.is_test:
            mask = load_image(mask_path, mask = True)
        
        if self.augmentation:
            image, mask = self.pad(image=image,mask=mask).values()
            image, mask = self.compose(image=image,mask=mask).values()
            mask = mask[:,:,np.newaxis]
            image, mask = ( torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32')), 
                             torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32')) )
        else:
            if not self.is_test:
                image, mask = self.pad(image=image,mask=mask).values()
                mask = mask[:,:,np.newaxis]
                image, mask = (torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32')), 
                               torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32')))
            else:
                image = self.pad(image=image)['image']
                image = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))
        
        
        if self.is_test:
            return (image,)
        else:
            if not self.binary:
                if self.classes is None:
                    return image, mask
                else:
                    if mask.sum() == 0:
                        target = torch.Tensor(1).fill_(0)
                    else:
                        target = torch.Tensor(1).fill_(1)
                    return image, mask, self.classes[index], target
            else:
                if mask.sum() == 0:
                    target = torch.Tensor(1).fill_(0)
                else:
                    target = torch.Tensor(1).fill_(1)
                return image, target
                
        
