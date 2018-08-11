# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:03:13 2018

@author: huijian
"""

import torch
import torch.nn as nn

def calculate_means(pred,gt,n_objects,max_n_objects,cuda=False,device=None):
    """
    pred: bs, height * width, n_filters torch.float32
    gt: bs,height*width,n_instances torch.int64
    """
    
    bs,n_loc,n_filters = pred.size()
    n_instances = gt.size(2)
    
    pred_repeated = pred.unsqueeze(2).expand(
            bs,n_loc,n_instances,n_filters) # (bs,n_loc,n_instances,n_filters)
    gt_expanded = gt.unsqueeze(3) # (bs,n_loc,n_instances,1)
    gt_expanded = gt_expanded.to(torch.float32)
    
    pred_masked = pred_repeated * gt_expanded
    
    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i,0]
        _pred_masked_sample = pred_masked[i,:,:_n_objects_sample] #(n_loc, n_objects, n_filters)
        _gt_expanded_sample = gt_expanded[i,:,:_n_objects_sample] #(n_loc,n_objects,1)

        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0) # n_objects, n_filters
        
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = max_n_objects - _n_objects_sample
            n_fill_objects = int(n_fill_objects)
            _fill_sample = torch.zeros(n_fill_objects, n_filters) # all 0
            if cuda:
                _fill_sample = _fill_sample.to(device)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
            
            
        means.append(_mean_sample)
    
    means = torch.stack(means) # (bs,max_n_objects,n_filters)
    return means

def calculate_variance_term(pred,gt,means,n_objects,delta_v,norm=2):
    """
    pred:(bs,height*width,n_filters)
    gt:(bs,height*width,n_instances)
    means:(bs,n_instances,n_filters)
    """
    
    bs, n_loc, n_filters =pred.size()
    n_instances = gt.size(2)
    
    means = means.unsqueeze(1).expand(bs,n_loc,n_instances,n_filters) # (bs,n_loc,n_instances,n_filters)
    pred = pred.unsqueeze(2).expand(bs,n_loc,n_instances,n_filters) # (bs, n_loc, n_instnaces,n_filters)
    gt = gt.unsqueeze(3).expand(bs,n_loc,n_instances,n_filters) # (bs,n_loc,n_instances,n_filters)
    # change the type from torch.ByteTensor to torch.FloatTensor
    gt = gt.to(torch.float32)
    
    _var = (torch.clamp(torch.norm((pred-means),norm,3) - delta_v, min=0.0)**2)*gt[:,:,:,0]
    
    var_term = 0.0
    for i in range(bs):
        _var_sample = _var[i,:,:n_objects[i,0]] # n_loc, n_objects
        _gt_sample = gt[i,:,:n_objects[i,0],0] # n_loc, n_objects
        
        var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term/bs
    return var_term

def calculate_distance_term(means, n_objects, delta_d, norm=2, cuda=False, device=None):
    """
    means:(bs,n_instances,n_filters)
    """
    
    bs, n_instances, n_filters = means.size()
    
    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = n_objects[i,0]
        _n_objects_sample = int(_n_objects_sample)
        
        if _n_objects_sample <=1:
            continue
        
        _mean_sample = means[i,:_n_objects_sample,:] # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(_n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1,0,2)
        
        diff = means_1 - means_2 # n_objects, n_objects, n_filters
        
        _norm = torch.norm(diff,norm,2)
        
        margin = 2*delta_d * (1.0 - torch.eye(_n_objects_sample))
        
        if cuda:
            margin = margin.to(device)
        
        _dist_term_sample = torch.sum(torch.clamp(margin-_norm, min=0.0)**2)
        _dist_term_sample = _dist_term_sample/(_n_objects_sample * (_n_objects_sample-1))
        
        dist_term += _dist_term_sample
    
    dist_term = dist_term/bs
    return dist_term

def calculate_regularization_term(means, n_objects, norm):
    """
    means:bs, n_instances, n_filters
    """
    bs, n_instances, n_filters = means.size()
    
    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i,:n_objects[i,0],:] # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term/bs
    
    return reg_term

def discriminative_loss(input, target, n_objects, max_n_objects, delta_v, delta_d, norm, cuda, device):
    """
    input: (bs, n_filters, fmap, fmap) (pred_instance_seg, torch.float32) 
    target: (bs, n_instances, fmap, fmap) (instance_seg_mask, torch.int64)
    n_objects: (bs,1)
    """
    alpha = 1.0
    beta = 1.0
    gamma = 0.001
    
    bs,n_filters, height,width = input.size()
    n_instances = target.size(1)
    
    input = input.permute(0,2,3,1).contiguous().view(bs,height*width,n_filters)
    target = target.permute(0,2,3,1).contiguous().view(bs,height*width, n_instances)
    
    if cuda:
        n_objects = n_objects.cpu()
    n_objects = n_objects.detach().numpy() # here n_objects become a numpy.array
    
    # 1.compute the cluster means
    cluster_means = calculate_means(input,target,n_objects,max_n_objects,cuda,device)
    # 2.compute the variance term
    var_term = calculate_variance_term(
            pred=input,gt=target,means=cluster_means,n_objects=n_objects,delta_v=delta_v,norm=norm)
    dist_term = calculate_distance_term(
            means=cluster_means,n_objects=n_objects,delta_d=delta_d,norm=norm,cuda=cuda,device=device)
    reg_term = calculate_regularization_term(means=cluster_means, n_objects=n_objects,norm=norm)
    
    loss = alpha * var_term + beta * dist_term + gamma * reg_term
    return loss

class DiscriminativeLoss(nn.Module):
    def __init__(self,delta_var, delta_dist, norm,
                 size_average=True,reduce=True,cuda=False,device=None):
        super(DiscriminativeLoss,self).__init__()
        
        self.reduce= reduce
        self.size_avaerage=size_average
        
        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = norm
        self.cuda = cuda
        self.device = device
    
    def forward(self,input,target,n_objects,max_n_objects):
        assert target.requires_grad == False
        # max_n_objects used here is the true value
        return discriminative_loss(input=input,target=target,n_objects=n_objects,max_n_objects=max_n_objects,
                                   delta_v = self.delta_var, delta_d = self.delta_dist,norm=self.norm,
                                   cuda = self.cuda, device=self.device)


