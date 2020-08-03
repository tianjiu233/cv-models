# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:23:57 2020

@author: huijianpzh
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou(pred,target,cls_average=True,ignore=None):
    
    b,c,h,w = pred.size()
    
    pred = F.softmax(pred,dim=1)
    pred = pred.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
    
    target = target.view(-1).unsqueeze(1)
    target_one_hot = None
    
    if ignore is None