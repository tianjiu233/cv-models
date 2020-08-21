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
from torch.utils.data import DataLoader

# model 
from model.UNet import Improved_UNet
from model.ResNetZoo import BasicBlock,Bottleneck
from model.ResNetZoo import ResNetUNet_wHDC
from model.SEZoo import ResNetUNet_wHDC_wSEConv
# dataset

from NAIPData import NAIPDataList,PrepareData
from GFChallenge import GFChallenge
from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug

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
    # load pre-train dataset  NAIP dataset    
    
    nlcd_key_txt = "nlcd_to_lr_labels.txt"
    nir = False
    cls_num = 5 # [0,1,2,3,4]
    mode = "npz"
    
    train_bs = 32
    val_bs = 10
    
    train_data_dir_list= [r"D:\repo\data\de_1m_2013\de_1m_2013_extended-train_patches",
                          r"D:\repo\data\ny_1m_2013\ny_1m_2013_extended-train_patches",]
    
    val_data_dir_list = [r"D:\repo\data\ny_1m_2013\ny_1m_2013_extended-val_patches",
                         r"D:\repo\data\de_1m_2013\de_1m_2013_extended-val_patches"]
    
    """
    train_data_dir_list = [r"/cetc/nas_remote_sensing/datasets/chesapeake_data/de_1m_2013_extended-train_patches",
                           r"/cetc/nas_remote_sensing/datasets/chesapeake_data/md_1m_2013_extended-train_patches",
                           r"/cetc/nas_remote_sensing/datasets/chesapeake_data/ny_1m_2013_extended-train_patches",
                           r"/cetc/nas_remote_sensing/datasets/chesapeake_data/pa_1m_2013_extended-train_patches",
                           r"/cetc/nas_remote_sensing/datasets/chesapeake_data/va_1m_2014_extended-train_patches",
                           r"/cetc/nas_remote_sensing/datasets/chesapeake_data/wv_1m_2014_extended-train_patches"]
    
    val_data_dir_list = [r"/cetc/nas_remote_sensing/datasets/chesapeake_data/de_1m_2013_extended-val_patches",
                         r"/cetc/nas_remote_sensing/datasets/chesapeake_data/md_1m_2013_extended-val_patches",
                         r"/cetc/nas_remote_sensing/datasets/chesapeake_data/ny_1m_2013_extended-val_patches",
                         r"/cetc/nas_remote_sensing/datasets/chesapeake_data/pa_1m_2013_extended-val_patches",
                         r"/cetc/nas_remote_sensing/datasets/chesapeake_data/va_1m_2014_extended-val_patches",
                         r"/cetc/nas_remote_sensing/datasets/chesapeake_data/wv_1m_2014_extended-val_patches"]
    """
    
    # train data
    train_data_transform = torchvision.transforms.Compose([PrepareData(0.5),
                                                           H_Mirror(),V_Mirror(),
                                                           Rotation(),ColorAug(),Nptranspose()])
    
    train_data = NAIPDataList(data_dir_list=train_data_dir_list,
                              nlcd_key_txt=nlcd_key_txt,
                              transform=train_data_transform,
                              mode=mode,nir=nir)
    
    # val data
    val_data_transform = torchvision.transforms.Compose([PrepareData(1),
                                                         Nptranspose()])
    
    val_data = NAIPDataList(data_dir_list=val_data_dir_list,
                              nlcd_key_txt=nlcd_key_txt,
                              transform=val_data_transform,
                              mode=mode,nir=nir)
    
    # data loader
    train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True)
    val_loader = DataLoader(dataset=val_data,batch_size=val_bs,shuffle=False)
    
     
    
    # define the model
    # net = Improved_UNet(in_chs=in_chs,cls_num=cls_num)
    # ------ ResNet34 ------
    # net = ResNetUNet_wHDC(in_chs=in_chs, out_chs=cls_num,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,5,7,9,17])
    net = ResNetUNet_wHDC_wSEConv(in_chs=in_chs, out_chs=cls_num,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,5,7,9,17])
    
    # ------ ResNet50 ------
    # net = ResNetUNet_wHDC_wSEConv(in_chs=in_chs, out_chs=cls_num,block=Bottleneck,layers=[3,4,6,3],rates=[1,2,5])
    
    # ------ ResNet101 ------
    # net = ResNetUNet_wHDC(in_chs=in_chs, out_chs=cls_num,block=Bottleneck,layers=[3,4,23,3],rates=[1,2,5])
    # net = ResNetUNet_wHDC_wSEConv(in_chs=in_chs, out_chs=cls_num,block=Bottleneck,layers=[3,4,23,3],rates=[1,2,5])
    
    net.apply(train_util.weights_init)
    trainer = Trainer(net,cuda=cuda,model_path=model_path)
    print(net)
    
    
    restore_model_name = "pre-train"
    restore_model = False
    if restore_model:
        trainer.restore_model(restore_model_name)
    
    # parameters for train
    epochs=int(10)
    loss_accu_interval = 1
    val_interval=1
    
    model_name = "pre-train"
    train_model = False
    if train_model:
        print("pre-train")
        trainer.train_model(train_loader,val_loader,
                            epochs=epochs,
                            loss_accu_interval=loss_accu_interval,
                            val_interval=val_interval,
                            model_name=model_name,
                            optim_mode="Adam")
    
    
    
    ### ------ Real Dataset ------
    # load the target data
    new_cls_num = 9
    GF_train_bs = 4
    GF_val_bs = 10
    # train
    data_dir = r"D:\repo\data\GF\Train"
    #data_dir = "/cetc/nas_remote_sensing/huijian/GF/Train"
    data_transform = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),
                                                     ColorAug(),
                                                     Nptranspose()])
    
    GFData_Train = GFChallenge(data_dir,scale_list = [128,192,256,512],
                               batch_interval = 10,transform = data_transform)
    # val 
    data_dir = r"D:\repo\data\GF\Val"
    # data_dir = "/cetc/nas_remote_sensing/huijian/GF/Val"
    data_transform = torchvision.transforms.Compose([Nptranspose()])
    GFData_Val = GFChallenge(data_dir,scale_list = [],
                               batch_interval = 10,transform = data_transform)
    
    
    GF_train_loader = DataLoader(dataset=GFData_Train,
                                 batch_size=GF_train_bs,shuffle=True,
                                 collate_fn = GFData_Train.collate_fn)
    GF_val_loader = DataLoader(dataset=GFData_Val,
                               batch_size=GF_val_bs,shuffle=False)
    
    # change the model(the special step for transfer-learning)
    # replace the net.classifer with a new conv layer
    trainer.net.classifier =nn.Conv2d(in_channels=64, out_channels=new_cls_num, kernel_size=1,stride=1,padding=0,bias=True)
    
    # restore the model
    restore_model_name = "GF-pre-train"
    restore_model = False
    if restore_model:
        trainer.restore_model(restore_model_name)
    
    # train model
    # parameters for train
    epochs=int(1e6)
    loss_accu_interval = 1
    val_interval = 1
    
    model_name = "GF-seg"
    train_model = True
    if train_model:
        trainer.train_model(GF_train_loader,GF_val_loader,
                            epochs=epochs,
                            loss_accu_interval=loss_accu_interval,
                            val_interval=val_interval,
                            model_name=model_name,
                            optim_mode="Adam")


