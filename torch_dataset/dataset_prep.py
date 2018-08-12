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
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape

    # Padding in needed for UNet models because they need image size to be divisible by 32 
    x_min_pad, x_max_pad, y_min_pad, y_max_pad = shape_image(height, width)
    
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
    else:
        img = img / 255.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
    
    
    
class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test = False, augmentation = False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        self.augmentation = augmentation
        
        #data augmentation
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.random_crop = transforms.RandomCrop(size = 128, pad_if_needed=True)
        self.random_hflip = transforms.RandomHorizontalFlip(p = 0.5)
        self.random_vflip = transforms.RandomVerticalFlip(p = 0.5)
        self.random_recrop = transforms.RandomResizedCrop(size = 128)
        self.random_rot = transforms.RandomRotation(degrees = (0,90))
        self.random_choice = transforms.RandomChoice([self.random_rot,self.random_hflip,self.random_vflip])       
        self.random_order = transforms.RandomOrder([self.random_rot,self.random_hflip,self.random_vflip])

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
            #concat for augmentation purposes
            image = torch.cat([image,mask], dim = 0)
       
        if self.augmentation:
            #to PIL image
            image = self.to_pil(image)
            
            #random choice
            image = self.random_choice(image)
            
            #to tensor
            image = self.to_tensor(image)
        
        #split back to mask and image
        if not self.is_test:
            image, mask = image[:-1,:,:], image[-1,:,:]
            mask = mask.unsqueeze(0)
        
        if self.is_test:
            return (image,)
        else:
            return image, mask
        
