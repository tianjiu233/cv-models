# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:37:57 2020

@author: huijian
"""

# band sequential: nir r g b

import os
import numpy as np

import matplotlib.pyplot as plt
from skimage import io

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader

# mylibs
from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug,Add_Mask

# [0,0,0] no-data
colormap_5 = [[0,0,0],
              [255,0,0],[0,255,0],
              [0,255,255],[255,255,0],
              [0,0,255]]

colormap_5 = np.array(colormap_5)

# [0,0,0] no-data
colormap_15=[[0,0,0],
             [200,0,0],[250,0,150],[200,150,150],
             [250,150,150],[0,200,0],[150,250,0],
             [150,200,150],[200,0,200],[150,0,250],
             [150,150,250],[250,200,0],[200,200,0],
             [0,0,200],[0,150,200],[0,200,250]
    ]

colormap_15 = np.array(colormap_15)

label2name_5={0:"no-data",
              1:"built-up",
              2:"farmland",
              3:"forest",
              4:"meadow",
              5:"water"
    }

label2name_15 = {0:"no-data",
                 1:"industrial land",
                 2:"urban residental",
                 3:"rural residential",
                 4:"traffic land",
                 5:"paddy field",
                 6:"irrigated land",
                 7:"dry cropland",
                 8:"garden plot",
                 9:"arbor",
                 10:"shrub land",
                 11:"natutal grassland",
                 12:"artificial grassland",
                 13:"river",
                 14:"lake",
                 15:"pond"
    }



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
    
class GID(Dataset):
    def __init__(self,data_dir,
                 transform = None,
                 mode="fine",nir=False):
        # mode can be coarse
        if nir == True:
            image_dir = "image_NirRGB"
        else:
            image_dir = "image_RGB"
        
        if mode == "fine":
            mode_dir = "Fine_land-cover_Classification_15classes" 
            label_dir = "label_15classes"
            self.colormap = colormap_15
        else:
            mode_dir = "Large-scale_Classification_5classes"
            label_dir = "label_5classes"
            self.colormap = colormap_5
            
        self.lut = build_color2label_lut(colormap = self.colormap)
            
        self.transform = transform
        
        self.image_dir = data_dir + "/" + mode_dir +  "/" + image_dir
        self.label_dir = data_dir + "/" + mode_dir + "/" + label_dir
        
        self.data = []
        
        # save the items
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        image =self.image_dir + "/" + self.data[index] + ".tif"
        label = self.label_dir + "/" + self.data[index] + "_label.tif"
        
        image = io.imread(image) # [height,width,3 or 4] uint8
        image = image*1./255 
        label = io.imread(label) # [heitgh,width,3] another process will be done
        
        label = color2label(color_img=label, lut=self.lut) # [height,width]
        label = label[...,np.newaxis]
        
        # print(label.shape)
        
        sample = {}
        sample["image"] = image
        sample["label"] = label.astype(np.float32)
        
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
        
    def show_patch(self,index):
        
        assert self.transform is None
        
        sample = self.__getitem__(index)
        
        image,label = sample["image"],sample["label"]
        
        label = label[...,0]
        
        label = label.astype(np.uint8)
        new_label = self.colormap[label]
    
        
        """
        h,w = label.shape[0],label.shape[1]
        new_label = np.zeros((h,w,3)) 
        for i,cm in enumerate(self.colormap):
            new_label[label==i] = self.colormap[i]
        """
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
        image,label,mask = sample["image"],sample["label"],sample["mask"]
        
        image = image.transpose(1,2,0)*255
        image = image.astype(np.uint8)
        label = label.transpose(1,2,0)
        mask = mask.transpose(1,2,0)
        
        label = label[:,:,0]
        mask = mask[:,:,0]
        
        label = label.astype(np.uint8)
        new_label = self.colormap[label]
        
        fig,axs = plt.subplots(1,3)
        
        # print(image.shape)
        
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(new_label)
        axs[1].axis("off")
        axs[2].imshow(mask,cmap="gray")
        axs[2].axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return image,new_label,mask

if __name__ == "__main__":
    
    data_dir = r"D:/repo/data/GID"
    mode = "fine"
    
    data_transform = None
    GIDData_1 = GID(data_dir,transform = data_transform,mode="fine",nir=False)
    GIDData_1.show_patch(5)
    sample = GIDData_1[5]
    
    data_transform = torchvision.transforms.Compose([Rotation(),H_Mirror(),V_Mirror(),ColorAug(),Nptranspose(),Add_Mask()])
    GIDData_2 = GID(data_dir,transform=data_transform,mode="fine",nir=False)
    image,new_label,mask = GIDData_2.show_sample(index=5)
    
                
        