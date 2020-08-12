# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:36:18 2020

@author: huijianpzh
"""

import os
import numpy as np

import matplotlib.pyplot as plt
from skimage import io

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader

# mylibs
from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug,Add_Mask

label2name={0:"daolu",
            1:"jianzhu",
            2:"guanmu_and_shu",
            3:"caoping",
            4:"tudi",
            5:"shuiti",
            6:"jiaotonggongji",
            7:"butoushuidimian",
            8:"qita"
    }

colormap = [
    [255,255,0],
    [0,0,255],
    [0,128,0],
    [0,255,0],
    [128,128,128],
    [0,255,255],
    [255,0,255],
    [255,255,255],
    [0,0,0]
    ]

colormap = np.array(colormap)

def build_color2label_lut(colormap):
    lut = np.zeros(256*256*256,dtype=np.int32)
    for i,cm in enumerate(colormap):
        # 0: no-data
        lut[cm[0]*65536+cm[1]*256+cm[2]]=i
    
    return lut

def color2label(color_img,lut):
    """
    color_img: [height,weight,3]
    lut: 1-d np.array
    """
    label = color_img.astype("int32")
    indices = label[:,:,0]*65536+label[:,:,1]*256 + label[:,:,2]
    
    return lut[indices]

class GF4Test(Dataset):
    def __init__(self,data_dir):
        
        self.image_dir = data_dir
        data = []
        
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                data.append(item.split(".tif")[0])
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        
        image = self.image_dir + "/" + self.data[index] + ".tif"
        image = io.imread(image)
        
        image = image.astype(np.float32)
        image = image*1./255
        
        sample = {}
        sample["image"] = image
        sample["name"] = self.data[index]
        
        return sample
        
class GFChallenge(Dataset):
    def __init__(self,data_dir,transform=None):
        
        self.transform = transform
        
        self.colormap = colormap
        self.lut = build_color2label_lut(colormap = self.colormap)
        
        self.data_dir = data_dir
        self.image_dir = self.data_dir+"/" + "image"
        self.label_dir = self.data_dir+"/" + "label"
        
        data = []
        
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                data.append(item.split(".tif")[0])
        
        self.data = data
        
        return 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        
        image = self.image_dir + "/" + self.data[index] + ".tif"
        label = self.label_dir + "/" + self.data[index] + "_gt.png"
        
        image = io.imread(image)
        label = io.imread(label)
        
        # some basic process
        image = image*1./255
        label = label[:,:,0:3]
        label = color2label(color_img=label,lut=self.lut)
        label = label[...,np.newaxis]
        label = label.astype(np.float32)
        
        # print(self.data[index])
        
        sample = {}
        sample["image"] = image.astype(np.float32)
        sample["label"] = label.astype(np.float32)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def show_patch(self,index):
        
        assert self.transform is None
        
        sample  = self.__getitem__(index)
        image,label = sample["image"],sample["label"]
        
        label = label[...,0]
        
        label = label.astype(np.uint8)
        new_label = self.colormap[label]
        
        fig,axs = plt.subplots(1,2)
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(new_label)
        axs[1].axis("off")
        plt.suptitle(self.data[index])
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return
    
    def show_sample(self,index):
        
        assert self.transform is not None
        
        sample = self.__getitem__(index)
        
        image,label = sample["image"],sample["label"]
        image = image.transpose(1,2,0)*255
        image = image.astype(np.uint8)
        label = label.transpose(1,2,0)
        
        label = label[:,:,0]
        label = label.astype(np.uint8)
        new_label = self.colormap[label]
        
        fig,axs = plt.subplots(1,2)
        
        # print(image.shape)
        
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(new_label)
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
        
if __name__=="__main__":
    
    data_dir = r"D:\repo\data\GF\Train"
    
    data_transform = None
    GFData_1 = GFChallenge(data_dir,data_transform)
    GFData_1.show_patch(index=100)
    
    data_transform = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),ColorAug(),Nptranspose()])
    GFData_2 = GFChallenge(data_dir,data_transform)
    GFData_2.show_sample(index=100)
    
    sample = GFData_2[8] 
    mask = ("mask" in sample.keys())

    