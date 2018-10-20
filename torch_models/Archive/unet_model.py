# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:29:57 2018

@author: s144299
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

#unet model
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True),
            #nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size = (3,3), padding = (1,1)),
            nn.BatchNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True),
            #nn.Dropout(0.2),
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
        self.dropout = nn.Dropout(0.25)   
        
    def forward(self, x):
        x1 = self.in_conv(x)
        
        x1 = self.max_pool(x1)
        x1 = self.dropout(x1)
        x2 = self.double_conv1(x1)
        
        x2 = self.max_pool(x2)
        x2 = self.dropout(x2)
        x3 = self.double_conv2(x2)
        
        x3 = self.max_pool(x3)
        x3 = self.dropout(x3)
        x4 = self.double_conv3(x3)
        
        #upsampling
        x = self.up_conv1(x4, x3)
        x = self.dropout(x)
        x = self.up_conv2(x, x2)
        x = self.dropout(x)
        x = self.up_conv3(x, x1)
        x = self.dropout(x)
        
        #out
        x = self.out_conv(x)
        
        return self.sigmoid(x)
        
def get_model(n_channels, n_classes):
    model = UNet(n_channels, n_classes)
    if torch.cuda.is_available(): model.cuda()
    return model
    
