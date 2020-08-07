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

def _sample_inpaint(sample,cuda):
    return

def _sample_segmentation(sample,cuda):
    return

def train_context_inpainting(train_loader,net,net_optimizer,
                             cuda = False,rec_weight = 0.5,
                             coach=None,use_coach_mask=False):
    net.train()
    if coach is not None:
        coach.eval()
    
    for batch_idx,sample in enumerate(train_loader,0):
        net_optimizer.zero_grad()
        intputs_,masks,targets = _sample_inpaint(sample,cuda)
        
        if coach is not None:
            masks,_,__ = coach.forward(inputs_,alpha=100,use_coach = uses_coach_masks)
        
        outputs_1 = net(inputs_*masks)
        mse_loss = (outputs_1 - targets)**2
        mse_loss = -1*F.threshold(-1*mse_loss,-2,-2)
        loss_rec = torch.sum(mse_loss*(1-masks))/torch.sum(1-masks)
        
        if coach is not None:
            loss_con = troch.sum(mse_loss*masks)/torch.sum(masks)
        else:
            outputs_2 = net(inputs_*(1-masks))
            mse_loss = (outputs_2 - targets)**2
            mse_loss = -1*F.threshold(-1*mse_loss,-2,-2)
            loss_con = torch.sum(mse_loss*masks)/torch.sum(masks)
        
        total_loss = rec_weight*loss_rec + (1-rec_weight)*loss_con
        total_loss.backward()
        net_optimizer.step()
        
    return

def train_coach(train_loader,net,coach,coach_optimizer,cuda=False):
    
    coach.train()
    net.eval()
    
    for batch_idx,sample in enumerate(train_loader,0):
        coach_optimizer.zero_grad()
        
        inputs_,masks,targets = _sample_inpaint(sample,cuda)
        masks,mus,logvars = coach.forward(inputs_,alpha=1)
        
        outputs = net(input_*mask).detach()
        mse_loss = (outputs-targets)**2
        mse_loss = -1*F.threshold(-1*mse_loss, -2, -2)
        # why *3 ?
        loss_rec = torch.sum(mse_loss*(1-masks))/(3*torch.sum(1-masks))
        
        mus = mus.mean(dim=2).mean(dim=2)
        logvars = logvars.mean(dim=2).mean(dim=2)
        
        KLD = 0
        try:
            KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
        else:
            KLD = 0
        
        total_loss = 1-loss_rec + 1e-6*KLD
        
        total_loss.backward()
        coach_optimizer.step()
        
    return

