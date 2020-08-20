# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:48:27 2020

@author: huijianpzh
"""

import torch
import torch.nn as nn

from Plug_and_Play import SEConvLayer
from ResNetZoo import BasicBlock,Bottleneck
from ResNetZoo import UPCONV_BN_AC,CONV_BN_AC

class ResNetUNet_wHDC_wSEConv(nn.Module):
    def __init__(self,in_chs,out_chs,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,3,5,7,9]):
        
        if block.__name__ == "BasicBlock":
            assert layers[-1]*2 == len(rates)
        else:
            assert layers[-1] == len(rates)
            
        self.inplanes = 64
        super(ResNetUNet_wHDC_wSEConv,self).__init__()
        
        self.conv1 = nn.Conv2d(in_chs,64,
                               kernel_size=7,stride=1,padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.se = SEConvLayer(chs=64,reduction=16)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        expansion=block.expansion
        
        # self.inplanes = 64 (no change in layer1) || self.inplanes = 64 (change to 256)
        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.se1 = SEConvLayer(chs=64*expansion,reduction=16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)                                     
        # we use maxpool to replace the conv of stride equal to 2.
        # self.inplnaes = 64 (change to 128)  || self.inplanes = 256 (change to 512)
        self.layer2 = self._make_layer(block,128, layers[1])
        self.se2 = SEConvLayer(chs=128*expansion,reduction=16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # self.inplnaes = 128 (change to 256)  || self.inplanes = 512 (change to 1024)
        self.layer3 = self._make_layer(block,256, layers[2])
        self.se3 = SEConvLayer(chs=256*expansion,reduction=16)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        
        # layer4 with HDC
        if block.__name__ == "Bottleneck":
            print("Bottleneck!")
            layer4 = []
            for idx in range(0,layers[3]):
                # self.inplnaes = 256 (change to 512)  || self.inplanes = 512 (change to 1024)
                layer4.append(self._make_layer_wBottleneck(planes=512,dilation=rates[idx]))
        else:
            print("BasicBlock!")
            layer4 = []
            for idx in range(0,layers[3],2):
                layer4.append(self._make_layer_wBasicBlock(planes=512,
                                                           dilation=rates[idx],extra_dilation=rates[idx+1]))
        self.layer4 = nn.Sequential(*layer4)
        self.se4 = SEConvLayer(chs=512*expansion,reduction=16)
        
        # decoder
        reduction = int(4/expansion)
        self.upsample1 = nn.Sequential(CONV_BN_AC(512*expansion,512*expansion//reduction,kernel=1,stride=1,pad=0),
                                       UPCONV_BN_AC(512*expansion//reduction,512*expansion//reduction),
                                       CONV_BN_AC(512*expansion//reduction,256*expansion,kernel=1,stride=1,pad=0)
                                       )
        
        self.upsample2 = nn.Sequential(CONV_BN_AC(2*256*expansion,256*expansion//reduction,kernel=1,stride=1,pad=0),
                                       UPCONV_BN_AC(256*expansion//reduction,256*expansion//reduction),
                                       CONV_BN_AC(256*expansion//reduction,128*expansion,kernel=1,stride=1,pad=0)
                                       )
        
        self.upsample3 = nn.Sequential(CONV_BN_AC(2*128*expansion,128*expansion//reduction,kernel=1,stride=1,pad=0),
                                       UPCONV_BN_AC(128*expansion//reduction,128*expansion//reduction),
                                       CONV_BN_AC(128*expansion//reduction,64*expansion,kernel=1,stride=1,pad=0)
                                       )
        
        self.upsample4 = nn.Sequential(CONV_BN_AC(2*64*expansion,64*expansion//reduction,kernel=1,stride=1,pad=0),
                                       UPCONV_BN_AC(64*expansion//reduction,64*expansion//reduction),
                                       CONV_BN_AC(64*expansion//reduction,64,kernel=1,stride=1,pad=0)
                                       )
        
        self.final_conv = nn.Sequential(CONV_BN_AC(2*64,64),
                                        CONV_BN_AC(64,64))
        # classifier
        self.classifier = nn.Conv2d(in_channels=64, 
                                    out_channels=out_chs, kernel_size=1)
        
        return
    
    def _make_layer(self,block,planes,blocks,
                    stride=1,bias=False):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            # for downsample, dilation makes no difference.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                          kernel_size=1,stride=stride,
                          bias=bias),
                nn.BatchNorm2d(planes*block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    def _make_layer_wBottleneck(self,planes,dilation=1,bias=False):
        downsample = None
        if self.inplanes != planes*Bottleneck.expansion:
            # for downsample, dilation makes no difference.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*Bottleneck.expansion,
                          kernel_size=1,stride=1,dilation=1,
                          bias=bias),
                nn.BatchNorm2d(planes*Bottleneck.expansion)
                )

        layer=Bottleneck(self.inplanes,planes,
                         stride=1,downsample=downsample,dilation=dilation)
        self.inplanes = planes * Bottleneck.expansion
        return layer
    
    def _make_layer_wBasicBlock(self,planes,dilation,extra_dilation,bias=False):
        downsample = None
        if  self.inplanes != planes*BasicBlock.expansion:
            # for downsample, dilation makes no difference.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*BasicBlock.expansion,
                          kernel_size=1,stride=1,dilation=1,
                          bias=bias),
                nn.BatchNorm2d(planes*BasicBlock.expansion)
                )
        layer=BasicBlock(self.inplanes,planes,
                         stride=1,downsample=downsample,bias=bias,
                         dilation=dilation,extra_dilation=extra_dilation)
        self.inplanes = planes * BasicBlock.expansion
        return layer
    
    def _encode(self,input_tensor):
        
        features = []
        
        x = self.conv1(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool1(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.maxpool1(x)
        
        x = self.layer2(x)
        features.append(x)
        x = self.maxpool2(x)
        
        x = self.layer3(x)
        features.append(x)
        x = self.maxpool3(x)
         
        output_tensor = x
        return output_tensor,features
    
    def _decode(self,input_tensor,features):
        
        x = self.upsample1(input_tensor) # [b,512*expansion,h,w]->[b,256*expansion,h,w]
        tmp =features[3]
        x = torch.cat([x,tmp],1)
        
        x = self.upsample2(x)
        tmp = features[2]
        x = torch.cat([x,tmp],1)
        
        x = self.upsample3(x)
        tmp = features[1]
        x = torch.cat([x,tmp],1)
        
        x = self.upsample4(x)
        tmp = features[0]
        x = torch.cat([x,tmp],1)
        
        output_tensor = self.final_conv(x)
        return output_tensor
    
    def _bottom(self,input_tensor):
        return self.layer4(input_tensor)
    
    def _enhance(self,input_tensor,features):
        
        x,_ = self.se4(input_tensor)
        
        features[0],_ = self.se(features[0])
        features[1],_ = self.se1(features[1])
        features[2],_ = self.se2(features[2])
        features[3],_ = self.se3(features[3])
        
        output_tensor=x 
        return output_tensor,features
    
    def forward(self,input_tensor):
        
        x,features = self._encode(input_tensor)

        x = self._bottom(x)
        
        x_,features_ = self._enhance(x,features)
        
        x = self._decode(x_,features_)
        
        output_tensor = self.classifier(x)
        
        return output_tensor

if __name__ == "__main__":
    input_ = torch.rand((2,3,256,256))
    # resnet_unet_whdc = ResNetUNet_wHDC(in_chs=3,out_chs=5,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,3,5,7,9])
    resnet_unet_whdc_wseconv = ResNetUNet_wHDC_wSEConv(in_chs=3,out_chs=5,block=Bottleneck,layers=[3,4,6,3],rates=[1,2,5])
    with torch.no_grad():
        output_ = resnet_unet_whdc_wseconv(input_)
        print(output_.shape)