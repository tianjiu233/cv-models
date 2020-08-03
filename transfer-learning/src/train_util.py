# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:27:54 2020

@author: huijianpzh
"""

# the file includes model init, lr scheduler.
"""
The codes are borrowed from
1. https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
2. https://www.cnblogs.com/hizhaolei/p/11226146.html

"""

import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

import math

# torch.nn.Module.apply function will be considered when initializing

def weights_init(m):
    classname = m.__class__.__name__
    # for every Linear Layer in a model
    if classname.find("Linear")!=-1:
        # (1) apply a uniform distribution to the weights and a bias =0
        #m.weight.data.uniform_(0,0,1.0)
        #m.bias.data.fill_(0)
        # (2) kaiming init
        init.kaiming_uniform_(m.weight,a=math.sqrt(3))
        if m.bias is not None:
            fan_in,_=init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(m.bias,-bound,bound)
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Conv2d):
        init.kaiming_uniform_(m.weight,a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(m.bias,-bound,bound)
        
if __name__=="__main__":
    
    # init methods 
    model = torchvision.models.resnet18(pretrained = False)
    model.apply(weights_init)
    
    # (1) fintune with the layer weights frozen
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(512,10)
    optimizer = optim.SGD(model.fc.parameters(),lr=1e-2,momentum=0.9)
    # (2) fintune without the layer weights frozen
    # optimizing the parameters with different lr rate
    model.fc = nn.Linear(512,10)
    
    ignored_params = list(map(id,model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    
    # different layer with different lr rate
    params_list = [
        {"params":base_params,"lr":0.001},
        ]
    params_list.append(
        {"params":model.fc.parameters(),"lr":0.01}
        )
    
    optimizer = optim.SGD(params_list,
                          0.001)
    
    
    
    
    