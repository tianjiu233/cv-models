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
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from trainer import Trainer

from model.UNet import Improved_UNet
from model.ResNetZoo import BasicBlock,Bottleneck
from model.ResNetZoo import ResNetUNet_wHDC
from model.SEZoo import ResNetUNet_wHDC_wSEConv

import GFChallenge
from GFChallenge import GF4Test

from my_util import TTA


def write_xml(input_,output_,name,output_path):
    
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
    provider_node.appendChild(provider_val)
    author_node = doc.createElement("author")
    author_val = doc.createTextNode(author)
    author_node.appendChild(author_val)
    pluginname_node = doc.createElement("pluginname")
    pluginname_val = doc.createTextNode(pluginname)
    pluginname_node.appendChild(pluginname_val)
    pluginclass_node = doc.createElement("pluginclass")
    pluginclass_val = doc.createTextNode(pluginclass)
    pluginclass_node.appendChild(pluginclass_val)
    time_node = doc.createElement("time")
    time_val = doc.createTextNode(time_)
    time_node.appendChild(time_val)
    
    research_node.appendChild(version_node)
    research_node.appendChild(provider_node)
    research_node.appendChild(author_node)
    research_node.appendChild(pluginname_node)
    research_node.appendChild(pluginclass_node)
    research_node.appendChild(time_node)
    root_node.appendChild(research_node)
    
    
    # segmentation
    segmentation_node = doc.createElement("segmentation")
    resultfile_node = doc.createElement("resultfile")
    resultfile_val = doc.createTextNode(output_) # 输出结果名称 
    resultfile_node.appendChild(resultfile_val)
    
    segmentation_node.appendChild(resultfile_node)
    root_node.appendChild(segmentation_node)
    
    with open(output_path+"/"+name+".xml", "w", encoding="utf-8") as f:
        # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")
    
    
def seg2file_wtta(net,test_data,output_path,
                  cuda = False,device=None):
    
    
    tta = TTA(net=net,activate=False,cuda=cuda,device=device,mode="mean")
    
    for idx,sample in tqdm(enumerate(test_data,0)):
        
        input_ = sample["image"] # np.array [h,w,3]
        name = sample["name"]
        

        output_ = tta.fuse2pred(image=input_) # tensor [h,w,c]
        

        _,output_ = torch.max(output_,dim=2)
        
        output_ = output_.detach().numpy().astype(np.uint8)
        
        output_ =GFChallenge.colormap[output_]
        output_ = output_.astype(np.uint8)
        
        file_name =output_path +"/" + name+"_gt.png"
        io.imsave(file_name,output_)
        
        write_xml(input_ = name+".tif",
                  output_=name+"_gt.png",
                  name=name,
                  output_path=output_path)
        
    return

if __name__=="__main__":
   
    
    # get the path
    #input_path = sys.argv[1]
    #output_path = sys.argv[2]
    
    
    input_path = r"D:\repo\data\GF\Val\image"
    output_path = r"D:\output_path"
    
    
    print("get input_path:{}".format(input_path))
    print("get output_path:{}".format(output_path))
    
    # build the dataset    
    test_data = GF4Test(data_dir = input_path)
    
    print("data len is {}".format(len(test_data)))
    
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
    net = ResNetUNet_wHDC_wSEConv(in_chs=in_chs, out_chs=cls_num,block=BasicBlock,layers=[3,4,6,3],rates=[1,2,3,5,7,9])
    # net = Improved_UNet(in_chs=in_chs,cls_num=cls_num)
    
    # build the trainer
    #model_path = "/workspace/checkpoint"
    model_path ="../checkpoint"
    trainer = Trainer(net,cuda=cuda,model_path=model_path)
    
    # restore the model
    model_name = "GF.pkl"
    trainer.restore_model(model_name)
    
    # segmentation
    seg2file_wtta(net=trainer.net,
             test_data=test_data,
             output_path=output_path,
             cuda = cuda,device=device)
    
    
    