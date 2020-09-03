#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:08:35 2019

@author: huijian
"""

import os
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import random

from skimage import io
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_xml(xml_file):
    """
    return objects(a dict)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    
    for item in root.findall("object"):
        name = item.find("name").text
        xmin = int(item.find("bndbox").find("xmin").text)
        ymin = int(item.find("bndbox").find("ymin").text)
        xmax = int(item.find("bndbox").find("xmax").text)
        ymax = int(item.find("bndbox").find("ymax").text)
        
        objects.append({"name":name,"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax})
    
    if len(objects)>0:
        return objects
    else:
        return None

def pad2square(img,pad_value=0):
    c,h,w = img.shape
    dim_diff = np.abs(h-w)
    # upper/left padding and lower/right padding
    pad1,pad2 = dim_diff//2, dim_diff-dim_diff//2
    # determin the padding
    pad = (0,0,pad1,pad2) if h<=w else (pad1,pad2,0,0)
    # add padding
    """
    The padding size by which to pad some dimensions of input are described 
    starting from the last dimension and moving forward.
    """
    img = F.pad(img,pad,"constant",value=pad_value)
    
    return img,pad

def horizonal_flip(imgs,labels):
    # [-1] means column
    imgs = torch.flip(imgs,[-1])
    # 2 means cx
    labels[:,2] = 1-labels[:,2]
    return imgs,labels

class HardHat(torch.utils.data.Dataset):
    
    # img_shape is used for normalize
    def __init__(self,ann_path,img_path,file_name,
                 augment = False,
                 img_shape=512,
                 multi_scale=False):
        
        """
        augment: To determine whether the filp operation will be done.
        multi_scale: To determine whether the multi_scale strategy will be adopted in the collote_fcn.
        """
        
        
        self.ann_path = ann_path
        self.img_path = img_path
        self.file_name = []
        
        """
        It is supposed that all the images will be adjusted to a square.
        If the images are not all resized to the square ones, the hardhat.py file should be changed a lot.
        """
        
        # the parameters below are related to the multi_scale strategy.
        self.img_shape=img_shape
        self.min_img_shape = self.img_shape - 3*32
        self.max_img_shape = self.img_shape + 3*32
        self.multi_scale=multi_scale
        # batch_count is used to controll the multi_scale strategy.
        self.batch_count = 0
        
        # data augmentation
        self.augment = augment
        
        
        # cls_dict the hyparameters for the dataset.
        self.cls_dict = {"blue":1,"white":2,"yellow":3,"red":4,
                           "none":0,
                           1:"blue",2:"white",3:"yellow",4:"red",
                           0:"none"}
        
        with open(file_name,"r") as file:
            tmp = file.readlines()
            for item in tmp:
                self.file_name.append(item.strip())
        self.file_name.sort()
        
    def __getitem__(self,index):
        
        file_name = self.file_name[index]
        # img
        img = io.imread(self.img_path+file_name+".jpg")
        
        # to record the original image shape
        img_shape = np.zeros(2)
        img_shape[0] = img.shape[0]
        img_shape[1] = img.shape[1]
        
        # first, convert the image to tensor
        img = img.transpose(2,0,1) # from (h,w,c) ---> (c,h,w)
        img = torch.FloatTensor(img) # np.array->tensor
        # pad the image (we choose square here) and resize
        img,pad = pad2square(img,pad_value=0) # img
        _,padded_h,padded_w = img.shape
        
        """
        for F.interpolate fcn
        The input dimensions are interpreted in the form: 
        mini-batch x channels x [optional depth] x [optional height] x width.
        so the img should add one more dimension
        """
        # resize img and at the same time the dimension of img changes (c,h,w) -> (c,h',w') 
        # we use unsqueeze and squeeze for the reseaon of F.interpolate
        """
        The self.img_shape will change if collate_fcn
        """
        img = F.interpolate(img.unsqueeze(0),size=self.img_shape,mode="nearest",align_corners=None).squeeze(0)
        img = (img/255.).clamp(0,1)
        
        # label
        objects = parse_xml(self.ann_path+file_name+".xml")
        
        # consider a special case, no objects
        if objects is None:
            return img,None
        
        nums = len(objects)
        label = np.zeros(shape=(nums,6),dtype=float)
        # label[:,0] is left as zeros
        for idx in range(nums):
            # label(name)
            label[idx][1] = self.cls_dict[objects[idx]["name"]]
            # xmin
            label[idx][2] = objects[idx]["xmin"]
            # ymin
            label[idx][3] = objects[idx]["ymin"]
            # xmax 
            label[idx][4] = objects[idx]["xmax"]
            # ymax
            label[idx][5] = objects[idx]["ymax"]
            
        
        # convert from (xmin,ymin,xmax,ymax) to (center_x,center_y,height,width)
        # and the xywh will be betweeen 0-1.
        x1 =label[:,2] + pad[0]
        y1 =label[:,3] + pad[2]
        x2 =label[:,4] + pad[1]
        y2 =label[:,5] + pad[3]
        
        label[:,2] = ((x1+x2)/2.)/padded_w
        label[:,3] = ((y1+y2)/2.)/padded_h
        label[:,4] = (x2-x1)/padded_w
        label[:,5] = (y2-y1)/padded_h
        
        # the final label we provide with is of shape (detecion_num,6)
        # 6: img_id(1),cls_name(1),box_coordinate(4) (scaled (cx,cy,width,height))
        label = torch.FloatTensor(label)
        
        # augmentation
        if self.augment:
            if np.random.random()<0.5:
                img,label = horizonal_flip(img,label)
        
        return img,label
    
    def __len__(self):    
        return len(self.file_name)
    
    def collate_fn(self,batch):
        img_list,label_list = list(zip(*batch))
        
        # (1) process label
        # The special case is that there is no object in one picture.
        for idx,label in enumerate(label_list):
            if label is None:
                continue
            label[:,0]=idx # to identify the img_id
            
        # the dimesion of label is  (obj_num,6) 6:img_id(1) cls_name(1) (c_x,c_y,w,h)(4)
        labels = torch.cat(label_list,dim=0)
        # print(labels.shape)
        # select new image_shape every tenth batch
        if self.multi_scale and self.batch_count % 10==0:
            #print("img_shape will change from")
            #print("img_shape:{}".format(self.img_shape))
            self.img_shape = random.choice(range(self.min_img_shape,self.max_img_shape+1,32))
            #print("to img_shape:{}".format(self.img_shape))
        
        # (2) process image (stacking them together)
        imgs = torch.stack(img_list,dim=0)
        # print(imgs.shape)
        
        # (3) process batch_count
        self.batch_count +=1
        
        return imgs,labels
        
    
    def _vis_sample(self,index):
        
        img,label = self.__getitem__(index=index)
        
        # convert the img from tensor to numpy (0-255)
        img = img.numpy().transpose(1,2,0)
        img = (img*255).astype(np.int)
        
        if label is not None:
            # label: img_id(1),cls_name(1),box(4)
            # box: (c_x,c_y,w,h) (the box is scaled)
            label = label.numpy()
            label[:,2] = label[:,2] * self.img_shape
            label[:,3] = label[:,3] * self.img_shape
            label[:,4] = label[:,4] * self.img_shape
            label[:,5] = label[:,5] * self.img_shape
            
            xmin = label[:,2] - label[:,4]/2
            xmax = label[:,2] + label[:,4]/2
            ymin = label[:,3] - label[:,5]/2
            ymax = label[:,3] + label[:,5]/2
            
            label[:,2] = xmin
            label[:,3] = ymin
            label[:,4] = xmax
            label[:,5] = ymax
        
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0,1,20)]
        
        #plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        
        if label is not None:
            
            bbox_colors = random.sample(colors,len(self.cls_dict))
            
            for idx in range(len(label)):
                
                cls_name,xmin,ymin,xmax,ymax = label[idx][1:]
                
                box_w = xmax-xmin
                box_h = ymax-ymin
                
                color = bbox_colors[int(cls_name)]
                bbox = patches.Rectangle((xmin,ymin), width=box_w, height=box_h,
                                         linewidth=2,edgecolor=color,facecolor="none")
                ax.add_patch(bbox)
                plt.text(
                    xmin,ymin,
                    s=self.cls_dict[int(cls_name)],
                    color = "White",
                    verticalalignment="top",
                    bbox = {"color":color,"pad":0},
                    )
        
        plt.show()
        
        return
    
    
if __name__=="__main__":
    annotation_path = "../GDUT-HWD/Annotations/"
    img_path = "../GDUT-HWD/JPEGImages/"
    label_path = "../GDUT-HWD/labels/"
    trainval_path = "../GDUT-HWD/ImageSets/Main/trainval.txt"
    test_path = "../GDUT-HWD/ImageSets/Main/test.txt"
    
    
    hats = HardHat(ann_path = annotation_path,
                   img_path = img_path,
                   file_name = trainval_path,
                   img_shape = 768,
                   augment = True,
                   multi_scale = True)
    
    hats._vis_sample(index=123)
    hats_dataloader = DataLoader(dataset=hats,batch_size=4,shuffle=True,
                                 collate_fn=hats.collate_fn)
    
    
    for idx,(img,label) in enumerate(hats_dataloader):
        print("This is batch_{}".format(idx))

        if idx == 10:
            print(label.shape)
            print(label[:,0])
            print(label[:,1])
            break