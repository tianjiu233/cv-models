# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:42:52 2020

@author: huijianpzh
"""

import numpy as np
import torch
import torchvision

from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug,Add_Mask

if __name__=="__main__":

    
    # load pre-train dataset
    data_dir = r"D:/repo/data/GID"
    mode = "fine"
    Pretrain_GID = GID(data_dir,transform=data_transform,mode="fine",nir=False)
    data_transform = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),ColorAug(),Nptranspose(),Add_Mask()])


    # load pre-train data

    # train the model

    # change the model
    # replace the net.classifer with a new one

    # load the target data

    # train the model 



