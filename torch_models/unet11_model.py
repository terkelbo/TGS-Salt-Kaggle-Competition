# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:12:42 2018

@author: terke
"""

from torch import nn
import torch
from torchvision import models

def conv3x3(in_, out):
    """ Convolution with padding and kernel size 3"""
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    """ Helper for Conv + BatchNorm + Relu """
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    """ Conv with padding and relu followed by transposed convolution """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels), #has padding, no reduction in tensor shape
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), #basically doubles shape of image (stride 2)
            #shape change follows this formula -> (in - 1)*stride - 2*padding + kernel_size + output_padding
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, drop = 0.5, pretrained=False):
        """
            Easy setup for using pretrained VGG11 features in the encoder part of the network. Uses transposed conv in decoder part.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(drop)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1] 
        self.conv1 = self.encoder[0] #(3, 64)
        self.conv2 = self.encoder[3] #(64, 128)
        self.conv3s = self.encoder[6] #(128, 256)
        self.conv3 = self.encoder[8] #(256, 256)
        self.conv4s = self.encoder[11] #(256, 512)
        self.conv4 = self.encoder[13] #(512, 512)
        self.conv5s = self.encoder[16] #(512, 512)
        self.conv5 = self.encoder[18] #(512, 512)

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8) #(512, 512, 256)
        
        #concatenation starting
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8) #(256 + 512, 512, 256) center + conv5
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4) #(256 + 512, 512, 128) dec5 + conv4
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2) #(256 + 128, 256, 64) dec4 + conv3
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters) #(128 + 64, 128, 32) dec3 + conv2
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters) #(32 + 32, 32) dec2 + conv1

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1) #(32, num_classes) #dec1 kernel size 1
        
        if num_classes == 1:
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.drop(self.relu(self.conv1(x)))
        conv2 = self.drop(self.relu(self.conv2(self.pool(conv1))))
        conv3s = self.drop(self.relu(self.conv3s(self.pool(conv2))))
        conv3 = self.drop(self.relu(self.conv3(conv3s)))
        conv4s = self.drop(self.relu(self.conv4s(self.pool(conv3))))
        conv4 = self.drop(self.relu(self.conv4(conv4s)))
        conv5s = self.drop(self.relu(self.conv5s(self.pool(conv4))))
        conv5 = self.drop(self.relu(self.conv5(conv5s)))

        center = self.center(self.pool(conv5))

        dec5 = self.drop(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.drop(self.dec4(torch.cat([dec5, conv4], 1)))
        dec3 = self.drop(self.dec3(torch.cat([dec4, conv3], 1)))
        dec2 = self.drop(self.dec2(torch.cat([dec3, conv2], 1)))
        dec1 = self.drop(self.dec1(torch.cat([dec2, conv1], 1)))
        x_out = self.final(dec1)
        return self.out_act(x_out)
    
    
def get_model(num_classes, num_filters, drop = 0.5, pretrained = False):
    model = UNet11(num_classes, num_filters, drop, pretrained)
    if torch.cuda.is_available(): 
        model.cuda() 
    else: 
        model.cpu()
    return model