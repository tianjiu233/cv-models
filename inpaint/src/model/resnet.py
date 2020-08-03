# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:17:55 2020

@author: huijianpzh
"""


import math
import copy

import torch
import torch.nn as nn


def bilinear(mode="bilinear",scale_factor=2):
    "bilinear umsampling"
    return nn.