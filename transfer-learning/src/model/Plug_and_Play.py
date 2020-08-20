# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:12:43 2020

@author: huijian
"""


import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self,chs,reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=chs, out_features=chs//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=chs//reduction, out_features=chs,bias=False),
            nn.Sigmoid()
            )
    def forward(self,input_tensor):
        
        x = input_tensor
        
        b,c,_,__ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        fc_feature = y
        return x*y.expand_as(x),fc_feature
        
        
class SEConvLayer(nn.Module):
    def __init__(self,chs,reduction=16):
        super(SEConvLayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_fc = nn.Sequential(
            nn.Conv2d(in_channels=chs, out_channels=chs//reduction, 
                      kernel_size=1,stride=1,padding=0,
                      bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=chs//reduction,out_channels=chs,
                      kernel_size=1,stride=1,padding=0,
                      bias=False),
            nn.Sigmoid()
            )
        
    def forward(self,input_tensor):
        
        x = input_tensor
        
        b,c,_,__ = x.size()
        y = self.avg_pool(x).view(b,c,1,1)
        y = self.conv_fc(y)
        conv_feature = y
        return x*y,conv_feature
        

if __name__=="__main__":
    # se block
    sample = torch.rand((3,64,256,256))
    se_fc = SELayer(chs=64,reduction=16)
    se_conv = SEConvLayer(chs=64,reduction=16)
    with torch.no_grad():
        fc_out,fc_feature = se_fc(sample)
        conv_out,conv_feature = se_conv(sample)