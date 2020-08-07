# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:38:40 2020

@author: huijianpzh
"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://www.jianshu.com/p/30043bcc90b6
https://blog.csdn.net/zziahgf/article/details/83589973
I think both of them are wrong and I modify the second one.
"""

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,reduction="mean"):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        self.reduction = reduction
        
        if isinstance(alpha,(float,int)):
            # binary
            if self.alpha>1:
                raise ValueError("Not supported value, alpha should be smaller than 1.0")
            else:
                self.alpha = torch.Tensor([alpha,1-alpha])
        elif isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.alpha /= torch.sum(self.alpha)
        
    def forward(self,pred,target):
        """
        pred:[b,c,h,w] 
        target:[b,h,w] range:[0,k-1]
        """
        print("Here")
    
        b,c,h,w = pred.size()
        pred = F.softmax(pred,dim=1)
        target = target.unsqueeze(1)
        
        if self.alpha.device != pred.device:
            self.alpha = self.alpha.to(pred.device)
        
        # log_p [b,c,h,w]
        log_p  = torch.log(pred + 1e-10)
        # torch.gather(input = log_p, dim = 1, index = target)
        # torch.gather is acutally a mapping function. We want to get the prob of the certain pixel
        # and log_p shoud be with a format of [b,1,h,w]
        # [b,c,h,w] -> [b,1,h,w]
        log_p = log_p.gather(1,target)
        # probs [b,1,h,w]
        probs = torch.exp(log_p)
        
        tmp = self.alpha.view(1,self.alpha.size(0),1,1)
        alpha = tmp.expand_as(pred)
        alpha = alpha.gather(1,target)
        
        gamma = torch.tensor(self.gamma,device=pred.device)
 
        # gamma: tensor; probs: [b,1,h,w] ; alpha: [b,1,h,w]; log_p: [b,1,h,w] 
        # loss will be [b,1,h,w]
        loss = -1*alpha*torch.pow((1-probs),gamma) * log_p
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss

