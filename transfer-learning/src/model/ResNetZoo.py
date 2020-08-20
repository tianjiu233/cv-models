# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:44:00 2020

@author: huijianpzh

In the file, some models are stored.
In order, they are:
    1.ResNet
    2.ResUNet
    # related paper:
    # Understanding Convolution for Semantic Segmentation
    3.ResUNet_wHDC (ResNet34 only)
    # related paper: 
    # Vehicle Instance Segmentation from Aerial Image and Video 
    # Using a Multitask Learning Residual Fully Convolutional Network
    4.ResNetFCN 
    
reference:
    https://github.com/chenxi116/pytorch-deeplab/blob/master/deeplab.py
"""


import math
import copy

import torch
import torch.nn as nn

def conv1x1(in_chs,out_chs,
            stride=1,pad=0,
            bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_channels=in_chs,out_channels=out_chs,kernel_size=1,
                     stride=stride,padding=pad,bias=bias)

def upsample(scale_factor,mode="bilinear",align_corners=False):
    return nn.Upsample(scale_factor=scale_factor,mode=mode,align_corners=align_corners)

def conv3x3(in_chs,out_chs,
            stride=1,pad=1,
            bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels=in_chs,out_channels=out_chs, 
                     kernel_size=3,
                     stride=stride,padding=pad,bias=bias)


def CONV_BN(in_chs,out_chs,
            kernel=3,stride=1,
            dilation=1,pad=1,
            bias=False):
    op = nn.Sequential(nn.Conv2d(in_chhannels=in_chs,out_channels=out_chs,
                                 kernel_size=kernel,stride=stride,
                                 dilation=dilation,padding=pad,bias=bias),
                       nn.BatchNorm2d(out_chs),
                       )
    return op
    

def CONV_BN_AC(in_chs,out_chs,
            kernel=3,stride=1,
            dilation=1,pad=1,
            bias=False):
    op = nn.Sequential(nn.Conv2d(in_channels=in_chs,out_channels=out_chs,
                                 kernel_size=kernel,stride=stride,
                                 dilation=dilation,padding=pad,bias=bias),
                       nn.BatchNorm2d(out_chs),
                       nn.ReLU(),
                       )
    return op

def UPCONV_BN_AC(in_chs,out_chs,
                 kernel=2,stride=2,
                 dilation=1,
                 pad=0,output_pad=0,
                 bias=False):
    op = nn.Sequential(nn.ConvTranspose2d(in_channels=in_chs,out_channels=out_chs,
                                          kernel_size=kernel,stride=stride,
                                          padding=pad,output_padding=output_pad,
                                          dilation=dilation,bias=bias),
                       nn.BatchNorm2d(out_chs),
                       nn.ReLU()
        )
    return op

"""
dilation will be only 1.
It not easy to adjust the dilate rate within BasicBlock.
Notice: Though We provider the dilation here, it will not work.
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride=1,downsample=None,bias=False,
                 dilation=1,extra_dilation=1):
        super(BasicBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes,planes,
                               kernel_size=3,stride=stride,
                               dilation=dilation,padding=dilation,
                               bias=bias)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.conv2 = nn.Conv2d(planes,planes,
                               kernel_size=3,stride=1,
                               dilation=extra_dilation,padding=extra_dilation,
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample=downsample
        self.stride=stride
    
    def forward(self,input_tensor):
        
        identify = input_tensor
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identify = self.downsample(input_tensor)
        
        output_tensor = self.relu( x + identify )
        
        return output_tensor
"""
The Bottleneck is different from BasicBlock for the dilation can be set directly.
"""
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None,dilation=1,bias=False):
        super(Bottleneck,self).__init__()
        
        # 1x1 conv stride
        self.conv1 = nn.Conv2d(inplanes,planes,
                               kernel_size=1,
                               stride=stride,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv dilation
        self.conv2 = nn.Conv2d(planes,planes,
                               kernel_size=3,
                               dilation=dilation,padding=dilation,
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes,planes * 4,
                               kernel_size=1,bias=bias)
        self.bn3 = nn.BatchNorm2d(planes*4)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.downsample = downsample
        self.stride=stride
    def forward(self,input_tensor):
        
        identify = input_tensor
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            identify = self.downsample(input_tensor)
          
        output_tensor = self.relu(x + identify)
        return output_tensor
"""

------ ResNet ------

"""
# Defaulty, it is ResNet34
class ResNet(nn.Module):
    def __init__(self,in_chs,out_chs,block=BasicBlock,layers=[3,4,6,3]):
        self.inplanes = 64
        super(ResNet,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) 
        
        self.fc_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_pool = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Linear(512*block.expansion,out_chs)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
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
    
    def forward(self,input_tensor):
        
        x = self.conv1(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.fc_pool(x)
        x = x.view(x.size(0),-1)
        
        output_tensor = self.fc(x)
        
        return output_tensor


"""

------ ResNet-FCN ------

"""

class ResNetFCN(nn.Module):
    def __init__(self,in_chs,out_chs,block=BasicBlock,layers=[3,4,6,3]):
        self.inplanes = 64
        super(ResNetFCN,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        expansion = block.expansion
        
        self.upsamplex8 = upsample(scale_factor=8)
        self.convx8 = conv1x1(in_chs=128*expansion,out_chs=out_chs)
        self.upsamplex16 = upsample(scale_factor=16)
        self.convx16 = conv1x1(in_chs=256*expansion,out_chs=out_chs)
        self.upsamplex32 = upsample(scale_factor=32)
        self.convx32 = conv1x1(in_chs=512*expansion,out_chs=out_chs)
        
        return
    
    
    def forward(self,input_tensor):
        
        x,features = self._encode(input_tensor)
        x = self._bottom(x)
        x = self._decode(x,features)
        
        output_tensor = x
        
        return output_tensor
    
    def _encode(self,input_tensor):
        features = []
        
        x = self.conv1(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
       
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        output_tensor = x
        
        return output_tensor,features
    
    def _decode(self,input_tensor,features):
        
        x = self.upsamplex32(input_tensor)
        x = self.convx32(x)
        
        tmp = features[1]
        tmp = self.upsamplex16(tmp)
        tmp = self.convx16(tmp)
        x = x + tmp
        
        tmp = features[0]
        tmp = self.upsamplex8(tmp)
        tmp = self.convx8(tmp)
        x = x + tmp
        
        output_tensor = x
        
        return output_tensor
    
    def _bottom(self,input_tensor):
        return self.layer4(input_tensor)
    
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


"""

------ ResNetUNet ------

"""

# Defaultly, ResNetUNet is ResNet34UNet
class ResNetUNet(nn.Module):
    def __init__(self,in_chs,out_chs,block=BasicBlock,layers=[3,4,6,3]):
        self.inplanes = 64
        super(ResNetUNet,self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(in_chs,64,
                               kernel_size=7,stride=1,padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)                                       
        # we use maxpool to replace the conv of stride equal to 2.
        self.layer2 = self._make_layer(block,128, layers[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer3 = self._make_layer(block,256, layers[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        # Actually, it is the bottom
        self.layer4 = self._make_layer(block,512, layers[3])
        
        
        # decoder
        expansion=block.expansion
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
        return  output_tensor
    
    def _bottom(self,input_tensor):
        return self.layer4(input_tensor)
    
    def forward(self,input_tensor):
        
        x,features = self._encode(input_tensor)
        x = self._bottom(x)
        x = self._decode(x,features)
        
        output_tensor = self.classifier(x)
        
        return output_tensor

# Here, we use ResNet-34.
# But we can still use ResNet-101 etc al.

class ResNetUNet_wHDC(nn.Module):
    def __init__(self,in_chs,out_chs,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,3,5,7,9]):
        
        if block.__name__ == "BasicBlock":
            assert layers[-1]*2 == len(rates)
        else:
            assert layers[-1] == len(rates)
            
        self.inplanes = 64
        super(ResNetUNet_wHDC,self).__init__()
        
        self.conv1 = nn.Conv2d(in_chs,64,
                               kernel_size=7,stride=1,padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        # self.inplanes = 64 (no change in layer1) || self.inplanes = 64 (change to 256)
        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)                                       
        # we use maxpool to replace the conv of stride equal to 2.
        # self.inplnaes = 64 (change to 128)  || self.inplanes = 256 (change to 512)
        self.layer2 = self._make_layer(block,128, layers[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # self.inplnaes = 128 (change to 256)  || self.inplanes = 512 (change to 1024)
        self.layer3 = self._make_layer(block,256, layers[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        
        # layer4 with HDC
        if block.__name__ == "Bottleneck":
            layer4 = []
            for idx in range(0,layers[3]):
                # self.inplnaes = 256 (change to 512)  || self.inplanes = 512 (change to 1024)
                layer4.append(self._make_layer_wBottleneck(planes=512,dilation=rates[idx]))
        else:
            layer4 = []
            for idx in range(0,layers[3],2):
                layer4.append(self._make_layer_wBasicBlock(planes=512,
                                                           dilation=rates[idx],extra_dilation=rates[idx+1]))
        self.layer4 = nn.Sequential(*layer4)
        
        
        # decoder
        expansion=block.expansion
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
    
    def forward(self,input_tensor):
        
        x,features = self._encode(input_tensor)

        x = self._bottom(x)
        x = self._decode(x,features)
        
        output_tensor = self.classifier(x)
        
        return output_tensor    


if __name__=="__main__":
    input_ = torch.rand((2,3,256,256))
    
    " --- ResNet --- "
    resnet =  ResNet(in_chs=3,out_chs=5,block=Bottleneck,layers=[3,4,6,3])
    
    " --- ResNetFCN ---"
    resnet_fcn = ResNetFCN(in_chs=3,out_chs=2,block=Bottleneck,layers=[3,4,6,3])
    
    " --- ResNetUNet ---"
    # resnet_unet = ResNetUNet(in_chs=3,out_chs=5,block=BasicBlock,layers=[3,4,6,3])
    resnet_unet = ResNetUNet(in_chs=3, out_chs=5,block=Bottleneck,layers=[3,4,6,3])
    
    " --- ResNetUNet_wHDC ---"
    # resnet_unet_whdc = ResNetUNet_wHDC(in_chs=3,out_chs=5,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,3,5,7,9])
    resnet_unet_whdc = ResNetUNet_wHDC(in_chs=3,out_chs=5,block=Bottleneck,layers=[3,4,6,3],rates=[1,2,5])
    with torch.no_grad():
        # output_ = resnet(input_)
        # output_ = resnet_fcn(input_)
        # output_ = resnet_unet(input_)
        output_ = resnet_unet_whdc(input_)
        print(output_.shape)
        
        