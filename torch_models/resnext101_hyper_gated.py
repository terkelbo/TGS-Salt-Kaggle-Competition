# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:31:02 2018

@author: terke
"""
import pretrainedmodels
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


class ResNext(nn.Module):
    def __init__(self, num_classes=1, is_deconv=True):
        super(ResNext, self).__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.resnet_layers = list(pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained='imagenet').children())
        self.maxpool = self.resnet_layers[0][3]
        self.conv_in = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv_in.weight = self.resnet_layers[0][0].weight
        for param_a, param_b in zip(self.conv_in.parameters(), self.resnet_layers[0].parameters()):
            param_a.data = param_b.data
        self.bn_in = nn.BatchNorm2d(64)
        for param_a, param_b in zip(self.bn_in.parameters(), self.resnet_layers[0][1].parameters()):
            param_a.data = param_b.data
        self.relu = nn.ReLU(True)
        
        self.conv1 = nn.Sequential(self.conv_in,
                                   self.bn_in,
                                   self.relu)
        self.conv2 = self.resnet_layers[1]
        self.conv3 = self.resnet_layers[2]
        self.conv4 = self.resnet_layers[3]
        self.conv5 = self.resnet_layers[4]
        
        bottom_channel_nr = 2048
        
        
        self.classifier = nn.Conv2d(512, 1, kernel_size=(4,4))
        
        self.center1 = nn.Sequential(nn.Conv2d(bottom_channel_nr, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2))
        self.center2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        self.center_decode = DecoderBlockV2(512*2, 512 + 256, 512, is_deconv)
        
        self.dec5 = DecoderBlockV2(bottom_channel_nr + 512, 512, 64, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + 64, 256, 64, is_deconv)
        self.dec3 = DecoderBlockV2(512 + 64, 128, 64, is_deconv)
        self.dec2 = ConvRelu(256 + 64, 64)
        self.dec1 = ConvRelu(64 + 64, 64)
        
        self.final = nn.Conv2d(832, num_classes, kernel_size=1)
        self.drop = nn.Dropout(p=0.5)

        self.SE6 = SCSE(64, 16)
        self.SE7 = SCSE(64, 16)
        self.SE8 = SCSE(64, 16)
        self.SE9 = SCSE(64, 16)
        self.SE10 = SCSE(64, 16)
        self.SE11 = SCSE(64, 16)
        
        

    def forward(self, x):
        conv1 = self.conv1(x) #; print(conv1.size())
        conv2 = self.conv2(conv1) #; print(conv2.size())
        conv3 = self.conv3(conv2) #; print(conv3.size())
        conv4 = self.conv4(conv3) #; print(conv4.size())
        conv5 = self.conv5(conv4) #; print(conv5.size())

        center1 = self.center1(conv5) #; print(center1.size())
        center2 = self.center2(center1) #; print(center2.size())
        
        center = self.center_decode(torch.cat([center1,center2],1)) #; print(center.size())

        dec5 = self.SE6(self.dec5(torch.cat([center, conv5], 1))) #; print(dec5.size())
        dec4 = self.SE7(self.dec4(torch.cat([dec5, conv4], 1))) #; print(dec4.size())
        dec3 = self.SE8(self.dec3(torch.cat([dec4, conv3], 1))) #; print(dec3.size())
        dec2 = self.SE9(self.dec2(torch.cat([dec3, conv2], 1))) #; print(dec2.size())
        dec1 = self.SE10(self.dec1(torch.cat([dec2, conv1], 1))) #; print(dec1.size())
        
        classification = torch.sigmoid(self.classifier(self.pool(center2)))
        
        
        cat = torch.cat([
                    dec1,
                    dec2,
                    dec3,
                    F.interpolate(dec4, scale_factor = 2, mode='bilinear', align_corners = False),
                    F.interpolate(dec5, scale_factor = 4, mode='bilinear', align_corners = False),
                    F.interpolate(center, scale_factor = 8, mode='bilinear', align_corners = False)
                ],1)
    
        #gate
        cat = classification * cat
        


        return self.final(cat), classification
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            
    
def get_model(num_classes=1, is_deconv=True):
    model = ResNext(num_classes, is_deconv)
    if torch.cuda.is_available(): 
        model.cuda() 
    else: 
        model.cpu()
    return model
