#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:23:54 2018

@author: huijian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def weights_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        torch.nn.init.xavier_normal(m.weight.data)
        torch.nn.init.xavier_normal(m.bias.data)

def weights_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        torch.nn.init.normal(m.weight.data)
        torch.nn.init.normal(m.bias.data)

class SegNet(nn.Module):
    def __init__(self,input_dim=3, output_dim=1, features=[64,128,256,512,512]):
        super(SegNet, self).__init__()
        # Encoders
        self.encoder_1 = nn.Sequential(
                nn.Conv2d(input_dim,features[0],7,padding=3),
                nn.BatchNorm2d(features[0]),
                nn.ReLU(),
                nn.Conv2d(features[0],features[0],3,padding=1),
                nn.BatchNorm2d(features[0]),
                nn.ReLU(),
                nn.MaxPool2d((2,2),stride=(2,2),return_indices=True),
                )
        
        self.encoder_2 = nn.Sequential(
                nn.Conv2d(features[0],features[1],3,padding=1),
                nn.BatchNorm2d(features[1]),
                nn.ReLU(),
                nn.Conv2d(features[1],features[1],3,padding=1),
                nn.BatchNorm2d(features[1]),
                nn.ReLU(),
                nn.MaxPool2d((2,2),stride=(2,2),return_indices=True),
                )
        
        self.encoder_3 = nn.Sequential(
                nn.Conv2d(features[1],features[2],3,padding=1),
                nn.BatchNorm2d(features[2]),
                nn.ReLU(),
                nn.Conv2d(features[2],features[2],3,padding=1),
                nn.BatchNorm2d(features[2]),
                nn.ReLU(),
                nn.Conv2d(features[2],features[2],3,padding=1),
                nn.BatchNorm2d(features[2]),
                nn.ReLU(),
                nn.MaxPool2d((2,2),stride=(2,2),return_indices=True),
                )
        self.encoder_4 = nn.Sequential(
                nn.Conv2d(features[2],features[3],3,padding=1),
                nn.BatchNorm2d(features[3]),
                nn.ReLU(),
                nn.Conv2d(features[3],features[3],3,padding=1),
                nn.BatchNorm2d(features[3]),
                nn.ReLU(),
                nn.Conv2d(features[3],features[3],3,padding=1),
                nn.BatchNorm2d(features[3]),
                nn.ReLU(),
                nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)
                )
        self.encoder_5 = nn.Sequential(
                nn.Conv2d(features[3],features[4],3,padding=1),
                nn.BatchNorm2d(features[4]),
                nn.ReLU(),
                nn.Conv2d(features[4],features[4],3,padding=1),
                nn.BatchNorm2d(features[4]),
                nn.ReLU(),
                nn.Conv2d(features[4],features[4],3,padding=1),
                nn.BatchNorm2d(features[4]),
                nn.ReLU(),
                nn.MaxPool2d((2,2),stride=(2,2),return_indices=True)
                )
        # MaxUnpool2d
        self.unpool_1 = nn.MaxUnpool2d(2,stride=2)
        self.unpool_2 = nn.MaxUnpool2d(2,stride=2)
        self.unpool_3 = nn.MaxUnpool2d(2,stride=2)
        self.unpool_4 = nn.MaxUnpool2d(2,stride=2)
        self.unpool_5 = nn.MaxUnpool2d(2,stride=2)
        # Decoders
        self.decoder_5 = nn.Sequential(
                nn.Conv2d(features[4],features[4],3,padding=1),
                nn.BatchNorm2d(features[4]),
                nn.ReLU(),
                nn.Conv2d(features[4],features[4],3,padding=1),
                nn.BatchNorm2d(features[4]),
                nn.ReLU(),
                nn.Conv2d(features[4],features[3],3,padding=1),
                nn.BatchNorm2d(features[3]),
                nn.ReLU(),
                )
        self.decoder_4 = nn.Sequential(
                nn.Conv2d(features[3],features[3],3,padding=1),
                nn.BatchNorm2d(features[3]),
                nn.ReLU(),
                nn.Conv2d(features[3],features[3],3,padding=1),
                nn.BatchNorm2d(features[3]),
                nn.ReLU(),
                nn.Conv2d(features[3],features[2],3,padding=1),
                nn.BatchNorm2d(features[2]),
                nn.ReLU(),
                )
        self.decoder_3 = nn.Sequential(
                nn.Conv2d(features[2],features[2],3,padding=1),
                nn.BatchNorm2d(features[2]),
                nn.ReLU(),
                nn.Conv2d(features[2],features[2],3,padding=1),
                nn.BatchNorm2d(features[2]),
                nn.ReLU(),
                nn.Conv2d(features[2],features[1],3,padding=1),
                nn.BatchNorm2d(features[1]),
                nn.ReLU(),
                )
        self.decoder_2 = nn.Sequential(
                nn.Conv2d(features[1],features[1],3,padding=1),
                nn.BatchNorm2d(features[1]),
                nn.ReLU(),
                nn.Conv2d(features[1],features[0],3,padding=1),
                nn.BatchNorm2d(features[0]),
                nn.ReLU(),
                )
        self.decoder_1 = nn.Sequential(
                nn.Conv2d(features[0],features[0],3,padding=1),
                nn.BatchNorm2d(features[0]),
                nn.ReLU(),
                nn.Conv2d(features[0],output_dim,3,padding=1),
                nn.BatchNorm2d(output_dim),
                # Attention
                nn.Sigmoid()
                )
    def forward(self,x):
        # encode
        x,id1 = self.encoder_1(x)
        x,id2 = self.encoder_2(x)
        x,id3 = self.encoder_3(x)
        x,id4 = self.encoder_4(x)
        x,id5 = self.encoder_5(x)
        
        # decoder
        x = self.unpool_5(x,id5)
        x = self.decoder_5(x)
        x = self.unpool_4(x,id4)
        x = self.decoder_4(x)
        x = self.unpool_3(x,id3)
        x = self.decoder_3(x)
        x = self.unpool_2(x,id2)
        x = self.decoder_2(x)
        x = self.unpool_1(x,id1)
        x = self.decoder_1(x)
        
        return x
if __name__ == "__main__":
    segnet = SegNet(input_dim=3, output_dim=1, features=[64,96,128,256,256])
    segnet.apply(weights_normal)
    images = Variable(torch.rand(4,3,256,256))
    prediction = segnet(images)
        




        