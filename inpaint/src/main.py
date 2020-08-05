# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:54:17 2020

@author: huijianpzh
"""


import torch

import warnings
warnings.filterwarnings("ignore")

## fix seed
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)

if __name__=="__main__":
    print("Testing...")
    
    # build the dataset (segmentation dataloader and the inpaint dataloader)
    
    # some parameters for erase process
    # size of each block used to erase image
    erase_shape = [16,16]
    erase_count = 16
    rec_weight = 0.99