# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:54:17 2020

@author: huijianpzh
"""


import torch
import torch.optim as optim
from torch.utils import data
import torchvision
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# mylibs
from dataio import context_inpaint_data,segmentation_data

from data_util import Stat4Data
from data_util import Nptranspose,H_Mirror,V_Mirror,RandomCrop

from train import train_context_inpainting,train_coach
from train import train_stepbystep
from train import validate4inpaint
from train import _save_model,_restore_model
from train import _visualize_inpaint

from model.resnet import resnet18_encoderdecoder,resnet18_coach_vae

import warnings
warnings.filterwarnings("ignore")

## fix seed
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)

# mean RGB values of images
AerialImageDataset_mean_rgb = np.array([[103.60683725],[109.06976655],[100.39146181]])   
# standard deviation RGB values of images  
AerialImageDataset_std_rgb = np.array([[48.61960021],[44.44692765],[41.98457744]])   
AerialImageDataset_stats = np.array([AerialImageDataset_mean_rgb,
                                     AerialImageDataset_std_rgb])  
 
if __name__=="__main__":
    print("Run main fcn...")
    # 0. hyparameters
    cuda = torch.cuda.is_available()
    if cuda:
        device= torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    data_dir = r"C:\Users\huijian\Downloads\repo\data\AerialImageDataset"
    model_path = "../checkpoint"
    # 1. build the dataset (segmentation dataloader and the inpaint dataloader)
    image_dir = data_dir +"/" + "images"
    label_dir = data_dir + "/" + "gt"
    
    # some parameters for erase process
    # size of each block used to erase image
    erase_shape = [16,16]
    erase_count = 16
    crop_shape = [128,128]
    rec_weight = 0.99
    
    data_transform = torchvision.transforms.Compose([Nptranspose()])
    inpaint_train_data = context_inpaint_data(image_dir=image_dir,stats=AerialImageDataset_stats,
                                              erase_shape = erase_shape,erase_count = erase_count,
                                              rotate = 0.5, 
                                              resize=0.5,
                                              crop = True, crop_shape = crop_shape,
                                              transform=data_transform)
    
    inpaint_val_data = context_inpaint_data(image_dir=image_dir,stats=AerialImageDataset_stats,
                                              erase_shape = erase_shape,erase_count = erase_count,
                                              rotate = 0.5, 
                                              resize=0.5,
                                              crop = True, crop_shape = crop_shape,
                                              transform=data_transform)
    
    # 2. networks and their optimizers
    in_chs =3
    out_chs =3
    drop_ratio =0.75
    net = resnet18_encoderdecoder(in_chs,out_chs)
    coach = resnet18_coach_vae(in_chs,drop_ratio)
    
    if cuda:
        net = net.to(device)
        coach = coach.to(device)
    
    net_optimizer = optim.SGD(net.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
    coach_optimizer = optim.Adam(coach.parameters(),lr=3e-4)
    
    
    # 3. train the network (for inpaint problem) 
    # 3.0 restore the model
    restore_model = False
    net_name = "inpaint_net"
    coach_name = "inpaint_coach"
    if restore_model:
        net = _restore_model(model_name=net_name,model_path=model_path,
                             cuda=cuda,device=device)
        coach = _restore_model(model_name=coach_name,model_path=model_path,
                             cuda=cuda,device=device) 
    # 3.0 prepare dataloader
    inpaint_train_dataloader = data.DataLoader(dataset = inpaint_train_data,
                                               batch_size=4,shuffle=True)
    inpaint_val_dataloader = data.DataLoader(dataset = inpaint_val_data,
                                             batch_size=1,shuffle=False)

    
    """   
    # 3.1 train model using train_context_inpainting & train_coach
    # ---begin---
    epochs = [100, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    # for the dataset we use here is not a big one.
    epochs = epochs*10
    lrs = [[1e-1, 1e-2, 1e-3, 1e-4],
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5], 
       [1e-5, 1e-5, 1e-5, 1e-5]]
    
    for e_ in range(0,len(epochs)):
        for epcoh in range(epochs[e_]):
            train_coach(inpaint_train_dataloader,
                        net,coach,coach_optimizer,
                        cuda=cuda,device=device)
        
        for epoch in range(epochs[e_]):
            
            if epoch == 90:
                net_optimizer = optim.SGD(net.parameters(), lr=lrs[e_][3], momentum=0.9, weight_decay=5e-4)
            if epoch == 80:
                net_optimizer = optim.SGD(net.parameters(), lr=lrs[e_][2], momentum=0.9, weight_decay=5e-4)
            if epoch == 40:
                net_optimizer = optim.SGD(net.parameters(), lr=lrs[e_][1], momentum=0.9, weight_decay=5e-4)
            if epoch == 0:
                net_optimizer = optim.SGD(net.parameters(), lr=lrs[e_][0], momentum=0.9, weight_decay=5e-4)
            
            train_context_inpainting(inpaint_train_dataloader,net,net_optimizer,
                                     cuda = cuda,device=device,
                                     rec_weight = rec_weight,
                                     coach=coach,use_coach_masks=True)
            
            loss_rec,loss_con,_,__ = validate4inpaint(inpaint_val_dataloader,
                                                      coach,net,
                                                      cuda=cuda,device=device)
            # save image
            _visualize_inpaint(inpaint_val_dataloader,
                       net=net,coach=coach,
                       stats = AerialImageDataset_stats,
                       prefix = "epoch_"+str(e+1)+"_",
                       cuda=cuda,device=device,
                       pic_dir = "../temp")
    
    # ---end---
    """
    
    # 3.2 train the model using train_stepbystep
    epochs = int(1e5)
    
    for e in range(epochs):
        train_stepbystep(inpaint_train_dataloader,
                         net,coach,
                         net_optimizer,coach_optimizer,
                         cuda=cuda,device=device,
                         rec_weight=rec_weight)
        
        if (e+1)%10 == 0:
            loss_rec,loss_con,_,__ = validate4inpaint(inpaint_val_dataloader,
                                                      coach,net,
                                                      cuda=cuda,device=device)
            print("Epoch-{}(validation): loss_rec:{:.5f}; loss_con:{:.5f}")
            with open("train-info.txt","a") as file_handle:    
                file_handle.write("Epoch-{}(validation): loss_rec:{:.5f}; loss_con:{:.5f}")
                file_handle.write("\n")
            _save_model(coach,model_name="coach"+"_epoch_"+str(e+1),model_path = "../checkpoint",cuda=False,device=None)
            _save_model(net,model_name="net"+"_epoch_"+str(e+1),model_path = "../checkpoint",cuda=False,device=None)
            # save image
            _visualize_inpaint(inpaint_val_dataloader,
                       net=net,coach=coach,
                       stats = AerialImageDataset_stats,
                       prefix = "epoch_"+str(e+1)+"_",
                       cuda=cuda,device=device,
                       pic_dir = "../temp")