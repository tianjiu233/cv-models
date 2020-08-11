# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:17:55 2020

@author: huijianpzh
"""


import math
import copy

import torch
import torch.nn as nn

def conv3x3(in_chs,out_chs,stride=1,bias=False):
    return nn.Conv2d(in_channels=in_chs, out_channels=out_chs, 
                     kernel_size=3,
                     stride=stride,padding=1,bias=bias)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride
    
    def forward(self,input_tensor,mask=None):
        
        residual = input_tensor
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            residual = self.downsample(input_tensor)
        
        output_tensor = self.relu( x + residual )
        
        return output_tensor
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, 
                               kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(in_channels=planes,out_channels=planes,
                               kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes,planes * 4,
                               kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride=stride
        
    def forward(self,input_tensor):
        
        residual = input_tensor
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            residual = self.downsample(input_tensor)
          
        output_tensor = self.relu(x + residual)
        
        return output_tensor
        
class ResNet_coach_vae(nn.Module):
    def __init__(self,in_chs,block,layers,drop_ratio=0.75):
        self.inplanes = 64
        super(ResNet_coach_vae,self).__init__()
        torch.cuda.manual_seed(7)
        torch.manual_seed(7)
        self.drop_ratio = drop_ratio
        
        self.conv1 = nn.Conv2d(in_channels=in_chs,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        
        self.mu = nn.Conv2d(in_channels=512,out_channels=100,kernel_size=1,bias=True)
        self.std = nn.Conv2d(in_channels=512,out_channels=100,kernel_size=1,bias=True)
        
        self.pred = nn.Conv2d(100,1,kernel_size=1,bias=True)
        
        self.upsample = nn.Upsample(scale_factor = 16, mode = "nearest")
        
        self.sigmoid = nn.Sigmoid()  # [0,1]
        
        
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                          kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        
        return nn.Sequential(*layers)
        
    def reparameterize(self,mu,logvar):
        std = logvar.mul(0.5).exp_()
        # the original code line here is
        # eps = Variable(std.data.new(std.size()).normal_())
        eps = (torch.empty(size=std.size(),dtype=std.dtype).normal_())
        if mu.is_cuda:
            eps = eps.to(mu.device)
        eps.requires_grad_(True)
        return eps.mul(std).add_(mu)
    
    def get_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        mu = self.mu(x)
        logvar = self.std(x)
        
        z = self.reparameterize(mu,logvar)
        d = self.pred(z)
        
        return d,mu,logvar
    
    def forward(self,input_tensor,alpha =1,use_coach=True):
        
        features = None
        mu = None
        logvar = None
        
        if not use_coach:
            size_ = input_tensor.size()
            features = torch.rand(size=(size_[0],1,int(size_[2]/16),int(size_[3]/16)),
                                  requires_grad=True)
            if input_tensor.is_cuda:
                features = features.cuda()
        else:
            features,mu,logvar = self.get_feature(input_tensor)
        
        size_ = features.size()
        
        features = features.view(size_[0],size_[1],size_[2]*size_[3])
        
        # topk will return a tupe (values,indices)
        p,_ = features.topk(k=int(size_[2]*size_[3]*self.drop_ratio),dim=2)
        # get the last one, which has the minimal value.
        partitions = p[:,:,-1]
        partitions = partitions.unsqueeze(2).expand(size_[0],size_[1],size_[2]*size_[3])
        
        mask = self.sigmoid(alpha*(features-partitions))
        mask = mask.view(size_)
        
        if not self.training:
            mask = (mask>0.5).float()
        
        # this help the coach net get the mask of the same size as the input_tensor
        # get the mask and then upsample
        mask = self.upsample(mask)
        
        return mask,mu,logvar

def resnet18_coach_vae(in_chs,drop_ratio,**kwargs):
    model = ResNet_coach_vae(in_chs,block=BasicBlock, layers=[2,2,2,2],drop_ratio=drop_ratio,**kwargs)
    return model

class ResNet_EncoderDecoder(nn.Module):
    def __init__(self,in_chs,out_chs,block,layers):
        self.inplanes = 64
        super(ResNet_EncoderDecoder,self).__init__()
        
        self.conv1 = nn.Conv2d(in_chs,64,
                               kernel_size=7,stride=2,padding=3,bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,128, layers[1],stride=2)
        self.layer3 = self._make_layer(block,256, layers[2],stride=2)
        self.layer4 = self._make_layer(block,512, layers[3],stride=2)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=512*block.expansion,out_channels=512, 
                                          kernel_size=4,stride=2,padding=1,bias=False)
        self.bn_d1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, 
                                          kernel_size=4,stride=2,padding=1,bias=False)
        self.bn_d2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                                         kernel_size=4,stride=2,padding=1,bias=False)
        self.bn_d3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=4,stride=2,padding=1,bias=False)        
        self.bn_d4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                          kernel_size=4,stride=2,padding=1,bias=False)
        self.bn_d5 = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32,out_chs,kernel_size=3,padding=1,bias=True)
        self.tanh = nn.Tanh()
        
        # initialization for the model
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    # I think m.bias.normal_() also works here.
                    m.bias.data.normal_(0,math.sqrt(2. / n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                    
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                          kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    def encode(self,x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
    def decode(self,x):
        
        x = self.deconv1(x)
        x = self.bn_d1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.bn_d2(x)
        x = self.relu(x)
        
        x = self.deconv3(x)
        x = self.bn_d3(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.bn_d4(x)
        x = self.relu(x)
        
        x = self.deconv5(x)
        x = self.bn_d5(x)
        x = self.relu(x)
        
        x = self.classifier(x)
        x = self.tanh(x)
        
        return x
    def forward(self,input_tensor):
        e = self.encode(input_tensor)
        d = self.decode(e)
        return d

def resnet18_encoderdecoder(in_chs,out_chs,**kwargs):
    model = ResNet_EncoderDecoder(in_chs, out_chs, block=BasicBlock, layers=[2,2,2,2],**kwargs)
    return model

if __name__=="__main__":
    
    # coach_net
    coach_net = resnet18_coach_vae(in_chs=3,drop_ratio=0.75)
    sample = torch.rand(size=(3,3,512,512))
    mask,mu,logvar= coach_net(sample,alpha=1,use_coach=True)
    """
    with torch.no_grad():
        mask,mu,logvar= coach_net(sample,alpha=1,use_coach=True)
      """  
    # encoderdecoder_net
    net = resnet18_encoderdecoder(in_chs=3, out_chs=3)
    sample = torch.rand(size=(3,3,512,512))
    outputs = net(sample*mask).detach()
    """
    with torch.no_grad():
        output = encoderdecoder_net(sample)
        """