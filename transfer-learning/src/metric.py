# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:47:27 2020

@author: huijianpzh
"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://www.jianshu.com/p/30043bcc90b6
https://blog.csdn.net/zziahgf/article/details/83589973
"""
class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,size_average=True):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = True
        
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
        target:[b,1,h,w] range:[0,k-1]
        """
        print("Here")
        if pred.dim() > 2:
            # [b,c,h,w] ->[b,c,h*w]
            pred = F.softmax(pred,dim=1)
            pred = pred.view(pred.size(0),pred.size(1),-1)
        # [b,1,h,w] -> [b*h*w,1]
        target = target.view(-1,1) 
        
        if self.alpha.device != pred.device:
            self.alpha = self.alpha.to(pred.device)
        
        # log_p = [b,c,h*w]
        log_p  = torch.log(pred + 1e-10)
        # torch.gather(input = log_p, dim = 1, index = target)
        log_p = log_p.gather(1,target)
        log_p = log_p.view(-1,1)
        
        probs = torch.exp(log_p)
        
        alpha = self.alpha.gather(0,target.view(-1))
        
        if not self.gamma.device == pred.device:
            gamma = self.gamma.to(pred.device)
        
        loss = -1*alpha*torch.pow((1-probs),gamma) * log_p
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        
        return loss



"""
https://github.com/milleniums/High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch
https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/utils.py
Computes and stores the average and current value.
"""

def confusion_matrix(pred,target,cls_num):
    """
    both pred and target are 1-d np.array
    """
    mask = (target>=0) & (target<cls_num)
    conf_mat = np.bincount(cls_num*target[mask].astype(int)+pred[mask],minlength=cls_num**2).reshape(cls_num,cls_num)

    return conf_mat


def evalue(conf_mat):
    """
    # compute the evaluation for the net
    conf_mat.sum(axis=1) gt col
    conf_mat.sum(aixs=0) pred row
    
    Delta is added to avoid nan.
    Other method can also be adopted to avoid it.
    """
    
    DELTA = 1e-9
    
    # total accuracy
    accu = np.diag(conf_mat).sum()/conf_mat.sum()
    # accuracy for per class
    accu_per_cls = np.diag(conf_mat)/(conf_mat.sum(axis=1) + DELTA)
    accu_cls = np.nanmean(accu_per_cls)
    
    # iou is a list for every class
    iou = np.diag(conf_mat)/(conf_mat.sum(axis=1)+conf_mat.sum(axis=0)-np.diag(conf_mat) + DELTA)
    mean_iou = np.nanmean(iou)
    # add FWIoU
    # https://blog.csdn.net/lingzhou33/article/details/87901365
    # some errors are in the blog, please to read the original paper
    fw_iou = (np.diag(conf_mat)*np.sum(conf_mat,axis=1))/(conf_mat.sum(axis=1)+conf_mat.sum(axis=0)-np.diag(conf_mat) + DELTA)
    fw_iou = fw_iou.sum()/conf_mat.sum()
    
    # kappa
    pe = np.dot(np.sum(conf_mat,axis=0),np.sum(conf_mat,axis=1))/(conf_mat.sum()**2)
    kappa = (accu - pe)/(1 - pe)
    return accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa