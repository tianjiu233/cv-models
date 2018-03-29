import torch
import torch.nn as nn
from torch.autograd import Variable

import copy

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        self.loss = self.criterion(input*self.weight, self.target)
        self.output = input
        return self.output
    
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
  
class GramMatrix(nn.Module):
    
    def forward(self, input):
        a, b, c, d = input.size()        
        features = input.view(a*b, c*d)
        G = torch.mm(features, features.t()) 
        return G.div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self,target,weight):
        super(StyleLoss, self).__init__()
        # self.target = target.detach()*weight
        # In the original version, the api .detach() was used to make
        # sure that requires_grad == False, Because it is required by criterion function.
        self.target = Variable(target.data.clone(),requires_grad=False)*weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self,input):
        
        self.output = input#.clone()
        self.G = self.gram(input.clone())
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output
    
    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers, style_layers,
                               style_weight=1000, content_weight=1,
                               ):

    content_losses = []
    style_losses = []
    
    cnn = copy.deepcopy(cnn)
    
    model = nn.Sequential()
    gram = GramMatrix()
    
    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_"+str(i)
            model.add_module(name, layer)
            
            if name in content_layers:
                target = model(content_img)#.clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_"+str(i),content_loss)
                content_losses.append(content_loss)
            
            if name in style_layers:
                target_feature = model(style_img)#.clone()
                target_gram = gram(target_feature)
                style_loss = StyleLoss(target_gram, style_weight)
                model.add_module("style_loss_"+str(i), style_loss)
                style_losses.append(style_loss)
        
        if isinstance(layer,nn.MaxPool2d):
            name = "pool_"+str(i)
            model.add_module(name,layer)
        
        if isinstance(layer,nn.ReLU):
            name = "relu_"+str(i)
            model.add_module(name,layer)
            
            if name in content_layers:
                target = model(content_img)#.clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss"+str(i),content_loss)
                content_losses.append(content_loss)
            
            
            if name in style_layers:
                target_feature = model(style_img)#.clone()
                target_gram = gram(target_feature)
                style_loss = StyleLoss(target_gram, style_weight)
                model.add_module("style_loss_"+str(i), style_loss)
                style_losses.append(style_loss)            
            i=i+1
            
    return model, style_losses, content_losses