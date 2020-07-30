# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:42:52 2020

@author: huijianpzh
"""

import numpy as np
import torch
import torchvision


from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug,Add_Mask
from GID import GID
from trainer import Trainer


if __name__=="__main__":

    # the shared parameters are listed here
    cuda = torch.cuda.is_available()
    model_path = "../checkpoint/"
    in_chs= 3
    ### Pre-Train
    
    # load pre-train dataset
    data_dir = r"D:/repo/data/GID"
    mode = "fine"
    data_transform = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),ColorAug(),Nptranspose(),Add_Mask()])
    Pretrain_GID = GID(data_dir,transform=data_transform,mode="fine",nir=False)
    
    
    # define the model
    cls_num=5
    net = ResNet34UNet(in_chs,cls_num)
    
    trainer = Trainer(net=net,cuda=cuda,model_path=model_path)
    
    model_name = "pre-train"
    
    restore_model = False
    if restore_model:
        trainer.restore_model(model_name)
    
    # parameters for train
    epochs=int(1e6)
    train_batch=16
    val_batch=10
    loss_accu_interval = 1
    val_interval =10
    
    train_model = True
    if train_model:
        trainer.train_model(train_data,val_data,
                            train_batch,val_batch,
                            epochs=epochs,
                            loss_accu_interval=loss_accu_interval,
                            val_interval=val_interval,
                            model_name=model_name):
    
    
    
    # Real Data
    
    # change the model
    # replace the net.classifer with a new one
    cls_num = 9
    trainer.net.classifer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=cls_num, 
                                                  kernel_size=1,stride=1,padding=0,bias=True),
                                        )
    
    

    # load the target data

    # train the model 



