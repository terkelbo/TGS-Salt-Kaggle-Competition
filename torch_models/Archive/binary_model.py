# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:04:42 2018

@author: terke
"""


from torch import nn
import torch
from torchvision import models
from torch.nn import functional as F

class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x): 
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        x = cSE + sSE

        return x

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
    
class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Resnet34_binary(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4


        self.final = nn.Conv2d(512, num_classes, kernel_size=4)
        self.drop = nn.Dropout(p=0.5)
        self.drop2d = nn.Dropout2d(p=0.25)        
        #SE blocks
        self.SE1 = SCSE(64, 16)
        self.SE2 = SCSE(64, 16)
        self.SE3 = SCSE(128, 16)
        self.SE4 = SCSE(256, 16)
        self.SE5 = SCSE(512, 16)        
        
        if num_classes == 1:
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.drop2d(self.SE1(self.conv1(x)))
        conv2 = self.drop2d(self.SE2(self.conv2(conv1)))
        conv3 = self.drop2d(self.SE3(self.conv3(conv2)))
        conv4 = self.drop2d(self.SE4(self.conv4(conv3)))
        conv5 = self.drop2d(self.SE5(self.conv5(conv4)))

        return self.out_act(self.final(conv5))
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            
    
def get_model(num_classes=1, num_filters=32, pretrained=False, is_deconv=True):
    model = Resnet34_binary(num_classes, num_filters, pretrained, is_deconv)
    if torch.cuda.is_available(): 
        model.cuda() 
    else: 
        model.cpu()
    return model
    
