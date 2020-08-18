# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:42:52 2020

@author: huijianpzh
"""

# official libs
import os
import numpy as np

# torch lib
import torch
import torchvision
import torch.nn as nn

# model 
from model.UNet import Improved_UNet

# dataset
from GID import GID
from GFChallenge import GFChallenge
from data_util import RandomCrop,Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug
from data_util import GenerateMask

# train
from trainer import Trainer
import train_util

# GPU setting
# os.environ["CUDA_VISIBLE_DEVICE"] = "3"
# torch.cuda.set_device(3)

if __name__=="__main__":

    # (1)the shared parameters are listed here
    # cuda,in_chs,and model_path
    cuda = torch.cuda.is_available()
    model_path = "../checkpoint/"
    in_chs= 3

    
    ### ------ Pre-Train ------
    # load pre-train dataset
    mode = "coarse"
    nir = False # in_chs will be 3
    # train
    data_dir = r"D:/repo/data/customized_GID/Train"
    data_transform = torchvision.transforms.Compose([RandomCrop(512),
                                                     Rotation(),
                                                     H_Mirror(),V_Mirror(),
                                                     ColorAug(),
                                                     Nptranspose(),
                                                    ])# GenerateMask()])
    train_data = GID(data_dir,transform=data_transform,mode=mode,nir=nir)
    # val
    data_dir = r"D:/repo/data/customized_GID/Val"
    data_transform = torchvision.transforms.Compose([Nptranspose(),
                                                     ])#GenerateMask()])
    val_data = GID(data_dir,transform=data_transform,mode=mode,nir=nir)
    
    
    # define the model
    cls_num=6 # "no-data" are not included.

    net = Improved_UNet(in_chs=in_chs,cls_num=cls_num)
    
    
    net.apply(train_util.weights_init)
    trainer = Trainer(net,cuda=cuda,model_path=model_path)
    
    
    restore_model_name = "pre-train"
    restore_model = False
    if restore_model:
        trainer.restore_model(restore_model_name)
    
    # parameters for train
    epochs=int(1e6)
    train_batch=8
    val_batch=10
    loss_accu_interval = 2
    val_interval=1
    
    model_name = "pre-train"
    train_model = False
    if train_model:
        trainer.train_model(train_data,val_data,
                            train_batch,val_batch,
                            epochs=epochs,
                            loss_accu_interval=loss_accu_interval,
                            val_interval=val_interval,
                            model_name=model_name,
                            optim_mode="Adam")
    
    
    
    ### ------ Real Dataset ------
    # change the model(the special step for transfer-learning)
    # replace the net.classifer with a new conv layer
    new_cls_num = 9
    trainer.net.classifier =nn.Conv2d(in_channels=64, out_channels=new_cls_num, kernel_size=1,stride=1,padding=0,bias=True)
                                                
    
    # load the target data
    # train
    # data_dir = r"D:\repo\data\GF\Train"
    data_dir = "/cetc/nas_remote_sensing/huijian/GF/Train"
    data_transform = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),
                                                     ColorAug(),
                                                     Nptranspose()])
    GFData_Train = GFChallenge(data_dir,data_transform)
    # val 
    # data_dir = r"D:\repo\data\GF\Val"
    data_dir = "/cetc/nas_remote_sensing/huijian/GF/Val"
    data_transform = torchvision.transforms.Compose([Nptranspose()])
    GFData_Val = GFChallenge(data_dir,data_transform)
    
    # restore the model
    restore_model_name = "pre-train"
    restore_model = False
    if restore_model:
        trainer.restore_model(restore_model_name)
    
    # train model
    # parameters for train
    
    epochs=int(1e6)
    train_batch=8
    val_batch=10
    loss_accu_interval = 2
    val_interval = 1
    
    model_name = "seg"
    train_model = True
    if train_model:
        trainer.train_model(GFData_Train,GFData_Val,
                            train_batch,val_batch,
                            epochs=epochs,
                            loss_accu_interval=loss_accu_interval,
                            val_interval=val_interval,
                            model_name=model_name,
                            optim_mode="Adam")


