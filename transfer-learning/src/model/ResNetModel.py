# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:02:28 2020

@author: huijianpzh
"""

import torch
import torch.nn as nn

class ResBlockA1(nn.Module):
    def __init__(self,in_chs,out_chs,
                 stride=1,kernel_size=3, # stride can only be 2 or 1.
                 padding=1,bias=True):
        super(ResBlockA1,self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_chs,out_channels=out_chs, 
                      kernel_size=kernel_size,stride=stride,
                      padding=padding,dilation=1,
                      bias=bias),
            nn.BatchNorm2d(num_features=out_chs),
            nn.ReLU(),
            # no change
            nn.Conv2d(in_channels=out_chs,out_channels=out_chs, 
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=bias),
            nn.BatchNorm2d(num_features=out_chs),
            )
        
        if stride == 1 and in_chs==out_chs:
            self.identify = None
        else:
            self.identify = nn.Sequential(nn.Conv2d(in_channels=in_chs, out_channels=out_chs, 
                                                    kernel_size=1,stride=stride,dilation=1,padding=0,bias=bias),
                                          nn.BatchNorm2d(num_features=out_chs))
        
        self.relu = nn.ReLU()
        
    def forward(self,input_tensor):
        if self.identify is None:
            output_tensor = self.relu(self.residual(input_tensor)+input_tensor)
        else:
            output_tensor = self.relu(self.residual(input_tensor)+self.identify(input_tensor))
        return output_tensor
        

class ResBlockB1(nn.Module):
    def __init__(self,in_chs,mid_chs,out_chs,
                 kernel_size=3,stride=1,
                 padding=1,bias=True):
        super(ResBlockB1,self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_chs,out_channels=mid_chs,
                      kernel_size=1,
                      stride=1,padding=0,dialtion=1,
                      bias=bias),
            nn.BatchNorm2d(num_features=mid_chs),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_chs,out_channels=mid_chs,
                      kernel_size=kernel_size,
                      stride=stride,padding=padding,dialtion=1,
                      bias=bias),
            nn.BatchNorm2d(num_features=mid_chs),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_chs,out_channels=out_chs,
                      kernel_size=1,
                      stride=1,padding=0,dialtion=1,
                      bias=bias),
            nn.BatchNorm2d(num_features=out_chs),
            )
        
        if stride==1:
            self.identify = None
        else:
            self.identify = nn.Sequential(nn.Conv2d(in_channels=in_chs, out_channels=out_chs, 
                                                    kernel_size=1,stride=stride,dilation=1,padding=0,bias=bias),
                                          nn.BatchNorm2d(num_features=out_chs))
        
        self.relu = nn.ReLU()
    def forward(self,input_tensor):
        if self.identify is None:
            output_tensor = self.relu(self.residual(input_tensor)+input_tensor)
        else:
            output_tensor = self.relu(self.residual(input_tensor)+self.identify(input_tensor))
        return output_tensor

# convtranspose2d-bn-relu
# the default parameters will make the picture 2x
class StdUpConv(nn.Module):
    def __init__(self,in_chs,out_chs,
                 kernel_size=2,stride=2,
                 padding=0,output_padding=0,
                 dilation=1,bias=True):
        super(StdUpConv,self).__init__()
        
        self.op = nn.Sequential(nn.ConvTranspose2d(in_channels=in_chs, out_channels=out_chs, 
                                                   kernel_size=kernel_size,stride=stride,
                                                   padding=padding,output_padding=output_padding,
                                                   dilation=dilation,bias=bias),
                                nn.BatchNorm2d(num_features=out_chs),
                                nn.ReLU())
    def forward(self,input_tensor):
        return self.op(input_tensor)

class StdConv(nn.Module):
    def __init__(self,in_chs,out_chs,
                 kernel_size=3,stride=1,
                 padding=1,
                 dilation=1,bias=True):
        super(StdConv,self).__init__()
        self.op = nn.Sequential(nn.Conv2d(in_channels=in_chs,out_channels=out_chs,
                                          kernel_size=kernel_size,stride=stride,
                                          padding=padding,dilation=dilation,
                                          bias=bias),
                                nn.BatchNorm2d(num_features=out_chs),
                                nn.ReLU())
    def forward(self,input_tensor):
        return self.op(input_tensor)
    
"""
If possible, a fpn structure should be added here to improve its performance on multi-scale.
HDC module may be helpful. "Understanding Convolution for Semantic Segmentation"
"""
class ResNet34UNet(nn.Module):
    """
    We build the model based on the paper 
    "Vehicle Instance Segmentation from Aerial Image and Video Using a Multitask Learning Residual Fully Convolutional Network".
    """
    def __init__(self,in_chs,cls_num):
        super(ResNet34UNet,self).__init__()
        
        # encoder
        # preblocks
        """
        Do we need a preblocks part?
        """
        # conv1 64-d downsample x2
        """
        We should find the paper about "why large kernel matters?
        """
        # the auxiliary convoltion to save the details.
        self.conv_aux = nn.Sequential(StdConv(in_chs=in_chs,out_chs=64),
                                      StdConv(in_chs=64,out_chs=64),
                                      )
        
        # conv1 64-d downsample x4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chs, 
                                             out_channels=64, 
                                             kernel_size=7, stride=2, padding=3))
        
        
        
        # the maxpool are included in conv2 to complete a U-Net structure.
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # conv2 x3 64-d downsample x8
        self.conv2 = nn.Sequential(ResBlockA1(in_chs=64, out_chs=64),
                                   ResBlockA1(in_chs=64, out_chs=64),
                                   ResBlockA1(in_chs=64, out_chs=64),
                                   )
        # conv3 x4 128-d downsample x16
        self.conv3 = nn.Sequential(ResBlockA1(in_chs=64, out_chs=128,stride=2),
                                   ResBlockA1(in_chs=128, out_chs=128),
                                   ResBlockA1(in_chs=128, out_chs=128),
                                   ResBlockA1(in_chs=128, out_chs=128),
                                   )
        
        # conv4 x6 256-d downsample x32
        self.conv4 = nn.Sequential(ResBlockA1(in_chs=128, out_chs=256,stride=2),
                                   ResBlockA1(in_chs=256, out_chs=256),
                                   ResBlockA1(in_chs=256, out_chs=256),
                                   ResBlockA1(in_chs=256, out_chs=256),
                                   ResBlockA1(in_chs=256, out_chs=256),
                                   ResBlockA1(in_chs=256, out_chs=256),
                                   )
        
        # conv5 x3 512-d (act as the bottom) downsample x64
        self.conv5 = nn.Sequential(ResBlockA1(in_chs=256,out_chs=512,stride=2),
                                   ResBlockA1(in_chs=512,out_chs=512),
                                   ResBlockA1(in_chs=512,out_chs=512),)
        
        
        # Decoder
        """
        We use transposeconv2d to function as the upsample layer.
        """
        self.upconv5 = nn.Sequential(
            StdConv(in_chs=512,out_chs=256),
            #StdConv(in_chs=256,out_chs=256),
            StdUpConv(in_chs=256,out_chs=256),
            )
        
        self.upconv4 = nn.Sequential(
            StdConv(in_chs=512,out_chs=256),
            #StdConv(in_chs=256,out_chs=256),
            StdUpConv(in_chs=256,out_chs=128),
            )
        
        self.upconv3 = nn.Sequential(
            StdConv(in_chs=256,out_chs=128),
            #StdConv(in_chs=128,out_chs=128),
            StdUpConv(in_chs=128,out_chs=64)
            )
        
        self.upconv2 = nn.Sequential(
            StdConv(in_chs=128,out_chs=64),
            #StdConv(in_chs=64,out_chs=64),
            StdUpConv(in_chs=64,out_chs=64)
            )
        
        self.upconv1 = nn.Sequential(
            StdConv(in_chs=128,out_chs=64),
            #StdConv(in_chs=64,out_chs=64),
            StdUpConv(in_chs=64,out_chs=64)
            )
        
        """
        Actually, I think there is upposed to be a upconv layer here and the conv_aux should be abondoned
        """
        self.upconv_aux = nn.Sequential(
            StdConv(in_chs=128,out_chs=64),
            StdConv(in_chs=64,out_chs=64),
            #StdUpConv(in_chs=64,out_chs=64)
            )
        
        self.classifier = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=cls_num, 
                                                  kernel_size=1,stride=1,padding=0,bias=True),
                                        )
        
        
    def forward(self,input_tensor):
        
        features = []
        
        # Encoding
        # the first items use input_tensor as the input
        x = self.conv_aux(input_tensor)
        features.append(x)
        
        x = self.conv1(input_tensor)
        features.append(x)
        
        x = self.maxpool(x)
        x = self.conv2(x)
        features.append(x)
        
        x = self.conv3(x)
        features.append(x)
        
        x = self.conv4(x)
        features.append(x)
        
        # bottom
        x = self.conv5(x)
        
        # Decoding
        # upsample x64->x32
        x = self.upconv5(x)
        
        # upsample x32->x16
        tmp = features.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv4(x)
        
        # upsamlpe x16->x8
        tmp = features.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv3(x)
        
        # upsample x8->x4
        tmp = features.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv2(x)
        
        # upsample x4->x2
        tmp = features.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv1(x)
        
        # upsample x2->x1
        tmp = features.pop()
        x = torch.cat((x,tmp),1)
        x = self.upconv_aux(x)
        
        output_tensor = self.classifier(x)
        
        return output_tensor
    
    
if __name__=="__main__":
    in_chs = 3
    cls_num = 17
    net =ResNet34UNet(in_chs,cls_num)
    
    sample_input = torch.rand((3,3,256,256))
    
    with torch.no_grad():
        output_tensor = net(sample_input)
