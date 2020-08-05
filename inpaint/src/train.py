# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:51:12 2020

@author: huijianpzh
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim 
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def visualize(cols=3,net=None,coach=None,use_coach_masks=False):
    fig,axs = plt.subplots(nrows=4,ncoks=cols,figsize=(9,9))

def train_context_inpainting(epoch,net,net_optimizer,coach=None,use_coach_mask=False):
    return

def train_coach(epoch,net,coach,coach_optimizer):
    return

