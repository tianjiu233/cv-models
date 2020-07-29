# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:36:18 2020

@author: huijianpzh
"""

import os
import numpy as np

import matplotlib.pyplot as plt
from skimage import io

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader

# mylibs
from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug,Add_Mask


class GFChallenge(Dataset):
    def __init__(self,data_dir,transform=None):
        data =[]
        return 
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return