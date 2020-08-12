# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:44:51 2020

@author: huijianpzh
"""

import sys
import argparse
import xml
import numpy as np
from skimage import io
from xml.dom import minidom
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import DataLoader

from trainer import Trainer
from model.UNet import Improved_UNet
from model.ResNetModel import ResNet34UNet

import GFChallenge
from GFChallenge import GF4Test

def write_xml(input_,output_):
    
    provider = r"中电慧遥"
    author = r"huijian"
    pluginname = r"地物标注"
    pluginclass= r"标注"
    time_ = r"2020-07-2020-11"
    
    #1. 创建dom树对象
    doc = minidom.Document()

    #2. 创建根结点，并用dom对象添加根结点
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)
    
    # source
    source_node = doc.createElement("source")
    filename_node = doc.createElement("filename")
    filename_val = doc.createTextNode(input_) # 文件名称
    filename_node.appendChild(filename_val)
    origin_node = doc.createElement("origin")
    origin_val = doc.createTextNode('GF2/GF3')
    origin_node.appendChild(origin_val)
    
    source_node.appendChild(filename_node)
    source_node.appendChild(origin_node)
    root_node.appendChild(source_node)
    
    # research
    research_node = doc.createElement("research")
    version_node = doc.createElement("version")
    version_val = doc.createTextNode("4.0")
    version_node.appendChild(version_val)
    provider_node = doc.createElement("provider")
    provider_val = doc.createTextNode(provider)
    pluginname_node = doc.createElement("pluginname")
    pluginname_val = doc.createTextNode(pluginname)
    pluginclass_node = doc.createElement("pluginclass")
    pluginclass_val = doc.createTextNode(pluginclass)
    time_node = doc.createElement("time")
    time_val = doc.createTextNode(time_)
    
    
    # segmentation
    segmentation_node = doc.createElement("segmentation")
    resultfile_node = doc.createElement("resultfile")
    resultfile_val = doc.createTextNode(output_) # 输出结果名称 
    resultfile_node.appendChild(resultfile_val)
    
    segmentation_node.appendChild(resultfile_node)
    root_node.appendChild(segmentation_node)
    
    
    
def seg2file(net,test_data,output_path,
             cuda = False,device=None):
    
    # data_loader = DataLoader(val_data,batch_size=1,shuffle=False)
    
    for idx,sample in enumerate(test_data):
        
        input_ = sample["image"]
        name = sample["name"]
        
        input_ = input_.transpose(2,0,1)
        input_ = torch.FloatTensor(input_)
        input_ = input_.unsqueeze(0)
        
        if cuda:
            input_ = input_.to(device)
        
        with torch.no_grad():
            output_ = net(input_) # [1,c,h,w]
            # TTA method can be add here
            output_ = output_.squeeze(0) # [c,h,w]
            _,output_ = torch.max(output_,dim=0)
            
        output_ = output_.detach().cpu().numpy().astype(np.uint8)
        
        output_ =GFChallenge.colormap[output_]
        output_ = output_.astype(np.uint8)
        
        file_name =output_path +"/" + name+"_gt.png"
        io.imsave(file_name,output_)
            
    return

if __name__=="__main__":
   
    
   # get the path
    #input_path = sys.argv[1]
    #output_path = sys.argv[2]
    # print(input_path)
    # print(output_path)
    
    input_path = r"D:\repo\data\GF\Val\image"
    output_path = r"D:\output_path"
    
    
    # build the dataset    
    test_data = GF4Test(data_dir = input_path)
    
    # use cuda or not
    cuda = torch.cuda.is_available()
    device =None
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    
    # define the model and build the trainer 
    # define the model
    in_chs = 3
    cls_num = 9 # GF-changllenge data
    # net = ResNet34UNet(in_chs=in_chs,cls_num=cls_num)
    net = Improved_UNet(in_chs=in_chs,cls_num=cls_num)
    
    # build the trainer
    model_path = "../checkpoint"
    trainer = Trainer(net,cuda=cuda,model_path=model_path)
    
    # restore the model
    model_name = "seg_999.pkl"
    trainer.restore_model(model_name)
    
    # segmentation
    seg2file(net=trainer.net,
             test_data=test_data,
             output_path=output_path,
             cuda = cuda,device=device)
    
    
    