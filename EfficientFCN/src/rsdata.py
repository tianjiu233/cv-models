# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:01:30 2020

@author: huijianpzh
"""

import os
import random
import numpy as np

import matplotlib.pyplot as plt
from skimage import io

import torchvision
from torch.utils.data import Dataset,DataLoader



class Vaihingen(Dataset):
    def __init__(self,image_dir,label_dir,dsm_dir=None,cuda=False,device=None):
        return
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return

class Postdam(Dataset):
    def __init__(self,image_dir,label_dir,dsm_dir=None,cuda=False,device=None):
        return
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return