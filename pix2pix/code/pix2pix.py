#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:39:46 2018

@author: huijian
"""

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        init.normal(m.weight.data,0.0,0.02)
        init.normal(m.bias.data,0.0,0.02)
    elif classname.find("Linear")!=-1:
        init.normal(m.weight.data,0.0,0.02)
        init.normal(m.bias.data,0.0,0.02)
    elif classname.find('BatchNorm2d')!=-1:
        init.normal(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)

def CONV_BN_RELU(in_ch, out_ch, dropout = False, decoder=False):
    """
    in_ch: in_channels
    out_ch: out_channels
    """
    if decoder:
        conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch,
                                  kernel_size=4, stride=2, padding=1, dilation=1)
        relu = nn.ReLU()
    else:
        conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                         kernel_size=4, stride=2,padding=1, dilation=1)
        relu = nn.LeakyReLU(0.2)
    
    if dropout:
        layer = nn.Sequential(
                conv,
                nn.BatchNorm2d(out_ch,momentum=0.1),
                nn.Dropout2d(p=0.5),
                relu,
                )
    else:
        layer = nn.Sequential(
                conv,
                nn.BatchNorm2d(out_ch,momentum=0.1),
                relu,
                )
    return layer
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        self.encoder = []
        self.decoder = []
        
        filters = [3,64,128,256,512,512,512,512,512]
        
        # encoder
        self.encoder_1 = nn.Sequential(
                            nn.Conv2d(in_channels=filters[0], out_channels=filters[1],
                                      kernel_size=4, stride=2, padding=1, dilation=1),
                            nn.LeakyReLU(0.2))                
        self.encoder_2 = CONV_BN_RELU(in_ch=filters[1],out_ch=filters[2])
        self.encoder_3 = CONV_BN_RELU(in_ch=filters[2],out_ch=filters[3])
        self.encoder_4 = CONV_BN_RELU(in_ch=filters[3],out_ch=filters[4])
        self.encoder_5 = CONV_BN_RELU(in_ch=filters[4],out_ch=filters[5])
        self.encoder_6 = CONV_BN_RELU(in_ch=filters[5],out_ch=filters[6])
        self.encoder_7 = CONV_BN_RELU(in_ch=filters[6],out_ch=filters[7])
        self.encoder_8 = nn.Sequential(
                            nn.Conv2d(in_channels=filters[7], out_channels=filters[8],
                                      kernel_size=4, stride=2, padding=1, dilation=1),
                          nn.LeakyReLU(0.2),)
        
        # decoder
        self.decoder_8 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=filters[8], out_channels=filters[7],
                                               kernel_size=4, stride=2, padding=1, dilation=1),
                            nn.BatchNorm2d(filters[7],momentum=0.1),
                            nn.Dropout2d(0.5),
                            nn.ReLU(),)
        self.decoder_7 = CONV_BN_RELU(in_ch=filters[7]*2,out_ch=filters[6],
                                      decoder=True,dropout=True)
        self.decoder_6 = CONV_BN_RELU(in_ch=filters[6]*2,out_ch=filters[5],
                                      decoder=True,dropout=True)
        self.decoder_5 = CONV_BN_RELU(in_ch=filters[5]*2,out_ch=filters[4],
                                      decoder=True,dropout=False)
        self.decoder_4 = CONV_BN_RELU(in_ch=filters[4]*2,out_ch=filters[3],
                                      decoder=True,dropout=False)
        self.decoder_3 = CONV_BN_RELU(in_ch=filters[3]*2,out_ch=filters[2],
                                      decoder=True,dropout=False)
        self.decoder_2 = CONV_BN_RELU(in_ch=filters[2]*2,out_ch=filters[1],
                                      decoder=True,dropout=False)
        self.decoder_1 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=filters[1]*2, out_channels=filters[0],
                                              kernel_size=4,stride=2,padding=1,dilation=1),
                                              nn.Tanh())

    def forward(self,x):
        saved_encoder = []
        
        x = self.encoder_1(x)
        saved_encoder.append(x)
        x = self.encoder_2(x)
        saved_encoder.append(x)
        x = self.encoder_3(x)
        saved_encoder.append(x)
        x = self.encoder_4(x)
        saved_encoder.append(x)
        x = self.encoder_5(x)
        saved_encoder.append(x)
        x = self.encoder_6(x)
        saved_encoder.append(x)
        x = self.encoder_7(x)
        saved_encoder.append(x)
        x = self.encoder_8(x)
        
        x = self.decoder_8(x)
        x = torch.cat([x,saved_encoder[6]],1)
        x = self.decoder_7(x)
        x = torch.cat([x,saved_encoder[5]],1)
        x = self.decoder_6(x)
        x = torch.cat([x,saved_encoder[4]],1)
        x = self.decoder_5(x)
        x = torch.cat([x,saved_encoder[3]],1)
        x = self.decoder_4(x)
        x = torch.cat([x,saved_encoder[2]],1)
        x = self.decoder_3(x)
        x = torch.cat([x,saved_encoder[1]],1)
        x = self.decoder_2(x)
        x = torch.cat([x,saved_encoder[0]],1)
        x = self.decoder_1(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.c64 = nn.Sequential(
                nn.Conv2d(in_channels=6,out_channels=64,
                          kernel_size=4,stride=2,padding=1),
                nn.LeakyReLU(0.2),
                )
        self.c128 = nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=128,
                          kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128,momentum=0.1),
                nn.LeakyReLU(0.2),
                )
        self.c256 = nn.Sequential(
                nn.Conv2d(in_channels=128,out_channels=256,
                          kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(256,momentum=0.1),
                nn.LeakyReLU(0.2),
                )
        self.c512 = nn.Sequential(
                nn.Conv2d(in_channels=256,out_channels=512,
                          kernel_size=4,stride=1,padding=1),
                nn.BatchNorm2d(512,momentum=0.1),
                nn.LeakyReLU(0.2),
                )
        self.final = nn.Sequential(
                nn.Conv2d(in_channels=512,out_channels=1,
                          kernel_size=4,stride=1,padding=1),
                nn.Sigmoid())
        
    def forward(self,x):
        x = self.c64(x)
        x = self.c128(x)
        x = self.c256(x)
        x = self.c512(x)
        x = self.final(x)
        return x

if __name__=="__main__":
    if False:
        D = Discriminator()
        test_input = Variable(torch.randn(1,6,256,256))
        test_output = D(test_input)
        print("D:test_output.shape:{}".format(test_output.shape))
    
    if True:
        G = Generator()
        test_input = Variable(torch.randn(1,3,512,512))
        test_output = G(test_input)
        print("G:test_output.shape:{}".format(test_output.shape))
    
    