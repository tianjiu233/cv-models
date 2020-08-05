# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:02:45 2020

@author: huijianpzh
"""


import os
import numpy as np
import cv2

import torch
from torch.utils import data

# mean BGR values of images
mean_bgr = np.array([85.5517787014, 92.6691667083, 86.8147645556])     
# standard deviation BGR values of images  
std_bgr = np.array([32.8860206505, 31.7342205253, 31.5361127226])      


class context_inpaint_data(data.Dataset):
    def __init__(self,image_dir,
                 erase_shape = [16,16],erase_count = 16,
                 rotate = 0.5, 
                 resize=0.5, resize_scale = [0.6,0.8,1.2,1.5],
                 crop = True, crop_shape = [128,128],
                 transform=None):
        
        self.image_dir = image_dir
        
        self.transform = transform
        
        # parameters
        self.erase_shape = erase_shape
        self.erase_coutn = erase_count
        
        self.rotate = rotate
        
        self.resize = resize
        self.resize_scale = resize_scale
        
        self.crop = crop
        self.crop_shape = crop_shape
        
        self.data = []
        # save the items
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0])
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        
        image = self.image_dir + "/" + self.data[index] + ".tif"
        
        # read in image and label
        
        # process
        # data_type,range etc.
        
        # resize,crop,rotate etc.
        if self.resize:
            image = self.get_random_crop(image, crop_shape=self.crop_shape)
        
        
        if self.rotate:
            choice = torch.LongTensor(1).random_(0,4)[0]
            angles = [0,90,180,270]
            angle = angles[choice]
            center = tuple(np.array(image.shape)[:2]/2)
            rot_mat = None
            rot_mat = cv2.getRotationMatrix2D(center,angle,1)
            image = cv2.warpAffine(image,rot_mat,image.shape[:2],flags=cv2.INTER_LINEAR)
        
        # erase
        mask = np.zeros(shape=image.shape[0:2],dtype=image.dtype)
        if self.erase_count == 1:
            offset = (image.shape[0]-self.earse_shape[0])/2
            end = offset + self.erase_shape[0]
            mask[offset:end,offset:end,:] = 0
        else:
            for c_ in range(self.erase_count):
                row =  torch.LongTensor(1).random_(0,image.shape[0]-self.erase_shape[0]-1)[0]
                col = torch.LongTensor(1).random_(0,image.shape[1]-self.erase_shape[1]-1)[0]
                
                mask[row:row+self.erase_shape[0],col:col+self.erase_shape[1],:]=0
                
        sample = {}
        sample["image"] = image
        sample["mask"] = mask
        
        # transform
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    def get_random_crop(self,image,crop_shape):
        """
        crop shape is of the format: [rows,cols]
        """
        r_offset = torch.LongTensor(1).random_(0,image.shape[0]-crop_shape[0]+1)[0]
        c_offset = torch.LongTensor(1).random_(0,image.shape[1]-crop_shape[1]+1)[0]
        
        if image.shape[0] == crop_shape[0]:
            r_offset = 0
        if image.shape[1] == crop_shape[1]:
            c_offset = 0
        
        crop_image = image[r_offset:r_offset+crop_shape[0],c_offset:c_offset+crop_shape[1],:]
        
        return crop_image


class segmentation_data(data.Dataset):
    def __init__(self,image_dir,label_dir,
                 transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        self.transform = transform
        
        self.data = []
        # save the items
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0])
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return
    
if __name__=="__main__":
    print("dataio.py testing...")