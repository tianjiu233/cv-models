# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:02:45 2020

@author: huijianpzh
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils import data

# mean BGR values of images
mean_bgr = np.array([85.5517787014, 92.6691667083, 86.8147645556])     
# standard deviation BGR values of images  
std_bgr = np.array([32.8860206505, 31.7342205253, 31.5361127226])      

"""
It shoulb be noticed that most oeprator in cv2 
requires the parameter in a order of (w,h) not (h,w).
"""

class context_inpaint_data(data.Dataset):
    def __init__(self,image_dir,
                 erase_shape = [16,16],erase_count = 16, # erase_shape [h,w]
                 rotate = 0.5, 
                 resize=0.5,
                 crop = True, crop_shape = [128,128], # crop_shape [h,w]
                 transform=None):
        
        self.image_dir = image_dir
        
        self.transform = transform
        
        # parameters
        self.erase_shape = erase_shape
        self.erase_count = erase_count
        
        self.rotate = rotate
        
        self.resize = resize
        
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
        if os.path.isfile(image):
            # cv2 load a image with the format (w,h,c) and a order of BGR
            # ---cv operator---
            image = cv2.imread(image) # [h,w,c], [B,G,R]
            b,g,r = cv2.split(image)
            image = cv2.merge([r,g,b])
        else:
            print("Coundn\'t find image:{}".format(image))  
        
        # np.random.random return a val in the half-open interval [0.0,1)
        # and np.random.randint also return a val with a half-open interval [low,high)
        # resize,crop,rotate etc.
        if np.random.random()>(1-self.resize):
            #image = self.get_random_crop(image, crop_shape=self.crop_shape)
            scales = [0.6,0.8,1.2,1.5]
            choice = np.random.randint(0,len(scales))
            scale = scales[choice] 
            image = cv2.resize(image,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            
        if self.crop:
            image = self.get_random_crop(image,self.crop_shape)
        
        if np.random.random()>=(1-self.rotate):
            angles = [0,90,180,270]
            choice = np.random.randint(0,len(angles))
            angle = angles[choice]
            w = round(image.shape[1]/2)
            h = round(image.shape[0]/2)
            # ---cv operator---
            # center [w,h]
            center = tuple([w,h])
            rot_mat = None
            rot_mat = cv2.getRotationMatrix2D(center,angle,1)
            # ---cv operator---
            # dsize [w,h]
            dsize = tuple([image.shape[1],image.shape[0]])
            image = cv2.warpAffine(image,rot_mat,dsize=dsize,flags=cv2.INTER_LINEAR)

        # erase
        mask = np.ones(shape=(image.shape[0], image.shape[1], 3), dtype = np.uint8)*255
        if self.erase_count == 1:
            offset =round((image.shape[0]-self.earse_shape[0])/2)
            end = offset + self.erase_shape[0]
        
            mask[offset:end,offset:end,:] = 0
        else:
            for c_ in range(self.erase_count):
                row = np.random.randint(0,image.shape[0] - self.erase_shape[0]+1)
                col = np.random.randint(0,image.shape[1] - self.erase_shape[1]+1)
                
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
        c_offset = np.random.randint(0,image.shape[0] - crop_shape[0]+1)
        r_offset = np.random.randint(0,image.shape[1] - crop_shape[1]+1)

        if image.shape[0] == crop_shape[0]:
            r_offset = 0
        if image.shape[1] == crop_shape[1]:
            c_offset = 0
        
        crop_image = image[r_offset:r_offset+crop_shape[0],c_offset:c_offset+crop_shape[1],:]
        return crop_image
    
    def show_patch(self,index):
        
        sample = self.__getitem__(index)
        image =sample["image"]
        mask = sample["mask"]
        
        if self.transform is not None:
            print("Special process..")
        
        fis,axs = plt.subplots(1,2)
        
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return image,mask


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
    
    data_dir = r"D:\repo\data\AerialImageDataset"
    
    image_dir = data_dir +"/" + "images"
    label_dir = data_dir + "/" + "gt"
    
    
    # (1) context_inpaint_data
    inpaint_dataset = context_inpaint_data(image_dir=image_dir,
                                           erase_shape = [16,16],erase_count = 16,
                                           rotate = 0.5, 
                                           resize=0.5,
                                           crop = True, crop_shape = [128,128],
                                           transform=None)
    inpaint_sample = inpaint_dataset[5]
    image = inpaint_sample["image"]
    mask = inpaint_sample["mask"]
    inpaint_dataset.show_patch(5)

    
    # 进行统计时，需要统计每个波段的平均值等，也需要统计类别的统计特性
    
    # (2) segmentation_data
    #segmentation_dataset = segmentation_data()