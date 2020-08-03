# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:44:53 2020

@author: huijian
"""


import torch
import torch.nn as nn
import torchvision


class BN_CONV_RELU(nn.Module):
    def __init__(self,in_chs,out_chs,
                 kernel_size=3,stride=1,padding=1,
                 dilation=1,bias=True):
        super(BN_CONV_RELU,self).__init__()
        self.op = nn.Sequential(nn.BatchNorm2d(num_features=in_chs),
                                nn.Conv2d(in_channels=in_chs,out_channels=out_chs,
                                kernel_size=kernel_size,stride=stride,padding=padding,
                                dilation=dilation,bias=bias),
                                nn.ReLU())
    def forward(self,input_tensor):
        return self.op(input_tensor)


class BN_UPCONV_RELU(nn.Module):
    def __init__(self,in_chs,out_chs,
                 kernel_size=2,stride=2,
                 padding=0,output_padding=0,
                 dilation=1,bias=True):
        super(BN_UPCONV_RELU,self).__init__()
        
        self.op = nn.Sequential(nn.BatchNorm2d(num_features=in_chs),
                                nn.ConvTranspose2d(in_channels=in_chs, out_channels=out_chs, 
                                                   kernel_size=kernel_size,stride=stride,
                                                   padding=padding,output_padding=output_padding,
                                                   dilation=dilation,bias=bias),
                                nn.ReLU())
        

    
    def forward(self,input_tensor):
        return self.op(input_tensor)

class Improved_UNet(nn.Module):
    def __init__(self,in_chs,cls_num,feats=[64,96]):
        super(Improved_UNet,self).__init__()
        
        # encoder structure
        # pre-conv
        self.pre = nn.Sequential(nn.Conv2d(in_channels=in_chs,out_channels=feats[0],
                                           kernel_size=3,stride=1,padding=1),
                                 nn.ReLU())
        
        # conv1
        self.conv1_1 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv1_2 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # conv2
        self.conv2_1 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv2_2 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv2_3 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # conv3
        self.conv3_1 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv3_2 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv3_3 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # conv4
        self.conv4_1 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv4_2 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv4_3 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # conv5
        self.conv5_1 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv5_2 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.conv5_3 = BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0])
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # bottom
        self.bottom = nn.Sequential(BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0]),
                                    BN_CONV_RELU(in_chs=feats[0],out_chs=feats[0]),
                                    BN_UPCONV_RELU(in_chs=feats[0],out_chs=feats[0]),
                                    )
        
        # decoder structure
        # upconv1
        self.upconv1 = nn.Sequential(BN_CONV_RELU(in_chs=2*feats[0], out_chs=feats[1]),
                                     BN_CONV_RELU(in_chs=feats[1], out_chs=feats[0]),
                                     BN_UPCONV_RELU(in_chs=feats[0], out_chs=feats[0]))
        # upconv2
        self.upconv2 = nn.Sequential(BN_CONV_RELU(in_chs=2*feats[0], out_chs=feats[1]),
                                     BN_CONV_RELU(in_chs=feats[1], out_chs=feats[0]),
                                     BN_UPCONV_RELU(in_chs=feats[0], out_chs=feats[0]))
        # upconv3
        self.upconv3 = nn.Sequential(BN_CONV_RELU(in_chs=2*feats[0], out_chs=feats[1]),
                                     BN_CONV_RELU(in_chs=feats[1], out_chs=feats[0]),
                                     BN_UPCONV_RELU(in_chs=feats[0], out_chs=feats[0]))
        # upconv4
        self.upconv4 = nn.Sequential(BN_CONV_RELU(in_chs=2*feats[0], out_chs=feats[1]),
                                     BN_CONV_RELU(in_chs=feats[1], out_chs=feats[0]),
                                     BN_UPCONV_RELU(in_chs=feats[0], out_chs=feats[0]))
        # upconv5
        self.upconv5 = nn.Sequential(BN_CONV_RELU(in_chs=2*feats[0], out_chs=feats[1]),
                                     BN_CONV_RELU(in_chs=feats[1], out_chs=feats[0]),
                                     nn.Conv2d(in_channels=feats[0], out_channels=cls_num, 
                                                   kernel_size=1,
                                                   stride=1,padding=0))
        
        # final_layer
        # self.final_layer = nn.Sequential(nn.Softmax2d())
        
        
    def forward(self,input_tensor):
        
        # encoding
        x = self.pre(input_tensor)
        feat_maps = []
        
        x = self.conv1_1(x)
        feat_maps.append(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        feat_maps.append(x)
        x = self.conv2_3(x)
        x = self.pool2(x)
                
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        feat_maps.append(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        feat_maps.append(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        feat_maps.append(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        
        x = self.bottom(x)
        
        # decoding
        
        tmp = feat_maps.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv1(x)
        
        tmp = feat_maps.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv2(x)
        
        tmp = feat_maps.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv3(x)
        
        tmp = feat_maps.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv4(x)
        
        tmp = feat_maps.pop()
        x = torch.cat((x,tmp),1)
        output_tensor = self.upconv5(x)
        
        #output_tensor = self.final_layer(output_tensor)
        
        return output_tensor
    
if __name__=="__main__":
    print("Testing")
    
    sample_input = torch.rand((3,3,512,512))
    
    net = Improved_UNet(in_chs=3,cls_num=2,feats=[64,96])
    
    with torch.no_grad():
        output_tensor = net(sample_input)
        print(output_tensor.shape)