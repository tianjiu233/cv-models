#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:14:14 2018

@author: huijian
"""

from skimage import io
from skimage import transform as sktrans

from torch.autograd import Variable
from torchvision import models

import torch
import torch.optim as optim
import torch.nn as nn

from utils import get_style_model_and_losses


def loader(img,img_size=512):
    """
    img:numpy.array
    """
    new_img = sktrans.resize(img,(img_size,img_size),mode="constant")
    new_img = torch.FloatTensor(new_img).unsqueeze(0).permute(0,3,1,2)
    return Variable(new_img)

def unloader(img,show=False):
    """
    img:torch.Tensor or Variable
    """
    if isinstance(img,Variable):
        img = img.data
    img = img.squeeze(0).permute(1,2,0).numpy()
    if show:
        io.imshow(img)
    return img
        

def prepare_data():
    
    style_file = "./data/picasso.jpg"
    content_file = "./data/dancing.jpg"
    
    style_img = io.imread(style_file)
    style_img = style_img[:,:,:]/255.
    
    
    content_img = io.imread(content_file)
    content_img = content_img[:,:,:]/255.
    
    style_img = loader(style_img)
    content_img = loader(content_img)
    
    input_img = content_img.clone()
    
    return style_img, content_img, input_img
    
    
def run_style_transfer(input_img,
                       style_img, content_img,
                       content_layers, style_layers,
                       style_weight=1000, content_weight=1,
                       epochs=300):
    cnn = models.vgg19(pretrained=True).features
    
    model, style_losses, content_losses = get_style_model_and_losses(
                                            cnn = cnn,
                                            content_layers = content_layers, style_layers = style_layers,
                                            style_img = style_img, content_img = content_img,
                                            style_weight=style_weight, content_weight=content_weight)
    
    if True:
        print(model)
    
    if True:
        print("Optimization Algorithm: LBFGS")
        input_param = nn.Parameter(input_img.data)
        optimizer=optim.LBFGS([input_param])
    
        run=[0]
        while run[0]<=epochs:
            def closure():
                style_score = 0
                content_score = 0
                
                optimizer.zero_grad()
                input_param.data.clamp_(0,1)
                model(input_param)
                
                for sl in style_losses:
                    style_score += sl.backward()
                for cl in content_losses:
                    content_score += cl.backward()
                
                run[0] += 1
                if run[0]%50==0:
                    print("run:{}".format(run))
                    print("style_loss:{:4f}, content_loss:{:4f}".format(
                            style_score.data.numpy()[0],content_score.data.numpy()[0]))
                return style_score+content_score
            optimizer.step(closure)
    
    if False: # another optimization algorithm
        print("Optimization Algorithm: Adam")
        input_param = nn.Parameter(input_img.data)
        optimizer=optim.Adam([input_param])
        
        run=[0]
        while run[0]<epochs:    
            style_score = 0
            content_score=0
        
            optimizer.zero_grad()
            input_param.data.clamp_(0,1)
            model(input_param)
            
            for sl in style_losses:
                style_score+=sl.backward()
            for cl in content_losses:
                content_score+=cl.backward()
        
            run[0]+=1
            if run[0]%50==0:
                print("run:{}".format(run))
                print("style_loss:{:4f}, content_loss:{:4f}".format(
                        style_score.data.numpy()[0],content_score.data.numpy()[0]))
        
            optimizer.step()
        
    input_param.data.clamp_(0,1)
    return input_param
    
if __name__ == "__main__":
    
    # identify the layers we want
    content_layers = ["conv_4"]
    style_layers = ["conv_1","conv_2","conv_3","conv_4","conv_5"]
    
    # prepare the data
    style_img, content_img, input_img = prepare_data()
    
    
    if False:
        unloader(style_img,show=False)
        unloader(content_img,show=True)
    
    print(input_img.size())

    
    result = run_style_transfer(input_img = input_img, 
                                content_img=content_img,
                                style_img = style_img,
                                content_layers=content_layers, 
                                style_layers=style_layers,
                                style_weight=1000, content_weight=1,
                                epochs=300)
    
    
    if True:
        unloader(result,show=True)
    
    