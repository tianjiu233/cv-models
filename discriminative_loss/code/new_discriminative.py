# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:26:59 2018

@author: huijian
"""

import torch
import torch.nn as nn

def calculate_means(pred,gt,n_objects,max_n_objects,cuda=False,device = None):
    """
    pred: (height*width,n_filters) torch.float32
    gt: (height*width,n_instances) torch.int64
    n_objects: int
    max_n_objects: int
    """
    n_loc,n_filters = pred.size()
    n_instances = gt.size(1)
    
    pred_repeated = pred.unsqueeze(1).expand(n_loc,n_instances,n_filters) # (n_loc,n_instances,n_filters)
    gt_expanded = gt.unsqueeze(2) # (n_loc,n_instances,1)
    gt_expanded = gt_expanded.to(torch.float32)
    
    pred_masked = pred_repeated * gt_expanded
    
    pred_masked_sample = pred_masked[:,:n_objects] # (n_loc,n_objects,n_filters)
    gt_expanded_sample = gt_expanded[:,:n_objects] # (n_loc,n_objects,1)
    
    mean_sample= pred_masked_sample.sum(0) / gt_expanded_sample.sum(0) # means for a batch
    
    if (max_n_objects - n_objects)!=0:
        n_fill_objects = max_n_objects - n_objects # int
        _fill_sample = torch.zeros(n_fill_objects, n_filters) # all 0
        if cuda:
            _fill_sample = torch.cat((mean_sample, _fill_sample), dim=0)
        mean_sample = torch.cat((mean_sample, _fill_sample), dim=0)
    
    return mean_sample # (max_n_objects, n_filters)

def calculate_variance_term(pred,gt,means,n_objects,delta_v,norm=2):
    """
    pred:(height*width,n_filters)
    gt:(height*width,n_instnaces)
    means:(n_instances,n_filters)
    """
    n_loc, n_filters = pred.size()
    n_instances = gt.size(1)
    
    means = means.unsqueeze(0).expand(n_loc,n_instances,n_filters) # (n_loc, n_instances, n_filters)
    pred = pred.unsqueeze(1).expand(n_loc,n_instances,n_filters) # (n_loc, n_instances, n_filters)
    gt = gt.unsqueeze(2).expand(n_loc,n_instances,n_filters) # (n_loc, n_instances, n_filters)
    # change the type from torch.ByteTensor to torch.FloatTensor
    gt = gt.to(torch.float32)
    
    var = (torch.clamp(torch.norm((pred-means),norm,2)- delta_v, min=0.0)**2)*gt[:,:,0]
    
    var_sample = var[:,:n_objects] # (n_loc,n_objects)
    gt_sample = gt[:,:n_objects,0] # (n_loc,n_objects)
    
    var_term = torch.sum(var_sample) / torch.sum(gt_sample)
    
    return var_term
    
    
def calculate_distance_term(means, n_objects, delta_d, norm=2, cuda=False, device=None):
    """
    means:(n_instances,n_filters)
    """
    
    n_instances,n_filters = means.size()
    
    _mean_sample = means[:n_objects,:] # (n_objects,n_filters)
    means_1 = _mean_sample.unsqueeze(1).expand(n_objects,n_objects,n_filters)
    means_2 = means_1.permute(1,0,2)
    
    diff = means_1 - means_2 # (n_objects, n_objects, n_filters)
    
    _norm = torch.norm(diff, norm, 2)
    
    margin = 2*delta_d * (1.0 - torch.eye(n_objects))
    
    if cuda:
        margin = margin.to(device)
    
    _dist_term_sample = torch.sum(torch.clamp(margin-_norm,min=0.0)**2)
    _dist_term_sample = _dist_term_sample/(n_objects*(n_objects-1))
    
    return _dist_term_sample

def calculate_regularization_term(means,n_objects,norm):
    """
    means:n_instances, n_filters
    """
    n_instances, n_filters = means.size()
    
    _mean_sample = means[:n_objects,:] # (n_objects,n_filters)
    _norm = torch.norm(_mean_sample, norm, 1) # (n_objects,)
    reg_term = torch.mean(_norm)
    
    return reg_term

def discriminative_loss(input,target,n_objects,max_n_objects,delta_v,delta_d,norm,cuda,device):
    """
    compute every the loss of every batch.
    input: (n_filters, fmap, fmap) (pred_instance_seg, torch.float32)
    target: (n_instances,fmap, fmap)
    n_objects: [1]
    """
    alpha = 1.0
    beta = 1.0
    gamma = 0.001
    
    n_filters, height, width = input.size()
    n_instances = target.size(0)
    
    input = input.permute(1,2,0).contiguous().view(height*width,n_filters)
    target = target.permute(1,2,0).contiguous().view(height*width,n_instances)
    
    # 1.compute the cluster means
    cluster_means = calculate_means(input, target, n_objects, max_n_objects, cuda, device)
    # 2.compute the variance term
    var_term = calculate_variance_term(
            pred=input,gt=target,means=cluster_means,n_objects=n_objects, delta_v=delta_v, norm=norm)
    if n_objects < 2:
        dist_term = 0
    else:
        dist_term = calculate_distance_term(
                means=cluster_means,n_objects=n_objects,delta_d=delta_d,norm=norm,cuda=cuda,device=device)
    reg_term = calculate_regularization_term(means=cluster_means,n_objects=n_objects,norm=norm)
    
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
        """
        input: (bs,n_filters,fmap,fmap) (pred_instance_seg, torch.float32)
        target: (bs,n_instances,fmap,fmap) (instance_seg_mask, torch.int64)
        n_objects: (bs,1)
        max_n_objects: int
        """
        assert target.requires_grad == False
        
        total_loss = 0
        
        # max_n_objects used here is the true value and we change the n_objects to numpy for simplification.
        if self.cuda:
            n_objects = n_objects.cpu()
        n_objects = n_objects.detach().numpy()
        
        for idx in range(len(n_objects)):
            sample_input = input[idx,:,:,:]
            sample_target = target[idx,:,:,:]
            sample_n_objects = int(n_objects[idx,0]) # it is better to use int instead of numpy.int32.
            
            if sample_n_objects>0:
                """
                sample_input: (n_filters,fmap,fmap)
                sample_target: (n_instances,fmap,fmap)
                sample_n_objets: it is supposed to be an int
                max_n_objects: int
                """
                sample_loss = discriminative_loss(input=sample_input, target=sample_target, 
                                                  n_objects = sample_n_objects, max_n_objects=max_n_objects,
                                                  delta_v=self.delta_var,delta_d=self.delta_dist,norm=self.norm,
                                                  cuda = self.cuda, device = self.device)
            else:
                sample_loss = 0
        
        total_loss = total_loss + sample_loss
        return total_loss
