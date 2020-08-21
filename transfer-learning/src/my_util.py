# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:03:31 2020

@author: huijianpzh
"""


import numpy as np
from skimage import transform
import torch
import torch.nn as nn

# my libs 
from model.UNet import Improved_UNet

# Test Time Augmentation for semantic segmentation
class TTA(object):
    def __init__(self,
                 net,
                 activate=True,
                 cuda=False,device=None,
                 mode="mean"):
        self.net = net
        self.cuda = cuda
        if self.cuda:
            self.device=device
        else:
            self.device=None
            
        self.mode = mode
        self.activate = activate
        
        return
    
    
    def _rotate(self,image,angle=90):
        r_ = transform.rotate(image,angle)
        r_ = self.forward(r_)        
        r_ = transform.rotate(r_,360-angle)
        return r_
    
    def _v_mirror(self,image):    
        v_ = np.flip(image,1).copy()
        v_ = self.forward(v_)
        v_ = np.flip(v_,1).copy()
        return v_
    
    def _h_mirror(self,image):
        h_ = np.flip(image,0).copy()
        h_ = self.forward(h_)
        h_ = np.flip(h_,0)
        return h_

    def forward(self,input_tensor):
        " input_tensor is np.array with a shape of [h,w,c]"       
        
        x = input_tensor.transpose(2,0,1)
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0) # [1,c,h,w]
        
        if self.cuda:
            x = x.to(self.device)
        
        with torch.no_grad():
            x = self.net(x)
            # to decide where to fuse
            if self.activate:
                x = nn.Softmax(dim=1)(x)
            
        x = x.detach().squeeze(0).cpu().numpy() # a tensor on cpu [1,cls,h,w]
        x = x.transpose(1,2,0)
        
        output_tensor = x # [h,w,cls]
        return output_tensor
    
    def fuse2pred(self,image):
        
        pred_ = self.forward(image)
        pred_ = pred_[...,np.newaxis]
        
        h_ = self._h_mirror(image)
        h_ = h_[...,np.newaxis]
        
        v_ = self._v_mirror(image)
        v_ = v_[...,np.newaxis]
        
        r90_ = self._rotate(image,angle=90)
        r90_ = r90_[...,np.newaxis]
        
        r180_ = self._rotate(image,angle=180)
        r180_ = r180_[...,np.newaxis]
        
        r270_ = self._rotate(image,angle=270)
        r270_ = r270_[...,np.newaxis]
        
        fusion = np.concatenate([pred_,h_,v_,r90_,r180_,r270_],axis=-1) # [6,h,w,cls_num]
        # turn to fusion to tensor to use api torch.max
        fusion = torch.FloatTensor(fusion)
        
        with torch.no_grad():
            if self.mode == "mean":
                fusion = torch.mean(fusion,dim=-1,keepdim=False) # [h,w,cls_num]
            elif self.mode == "max":
                fusion,_ = torch.max(fusion,axis=-1) # val/idx  [h,w,cls_num]
            
        return fusion # torch.tensor [h,w,c]
        
    
if __name__ == "__main__":
    
    input_ = torch.rand((256,256,3))
    input_ = input_.numpy()
    
    net = Improved_UNet(in_chs=3, cls_num=6)
    
    tta = TTA(net=net,activate=True,cuda=False,device=None,mode="max")
    
    pred = tta.fuse2pred(image=input_)
    