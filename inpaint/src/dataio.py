# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:02:45 2020

@author: huijianpzh
"""

# --- libs ---

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import torchvision

from data_util import Stat4Data
from data_util import Nptranspose,H_Mirror,V_Mirror,RandomCrop

# --- codes ---

# mean RGB values of images
AerialImageDataset_mean_rgb = np.array([[103.60683725],[109.06976655],[100.39146181]])   
# standard deviation RGB values of images  
AerialImageDataset_std_rgb = np.array([[48.61960021],[44.44692765],[41.98457744]])   
AerialImageDataset_stats = np.array([AerialImageDataset_mean_rgb,
                                     AerialImageDataset_std_rgb])  

"""
It shoulb be noticed that most oeprator in cv2 
requires the parameter in a order of (w,h) not (h,w).
"""


class context_inpaint_data(data.Dataset):
    def __init__(self,image_dir,suffix=".tif",stats=None,
                 erase_shape = [16,16],erase_count = 16, # erase_shape [h,w]
                 resize=0.5,
                 rotate = 0.5,
                 v_mirror=0.5,h_mirror=0.5,
                 crop = True, crop_shape = [128,128], # crop_shape [h,w]
                 transform=None):
        
        self.image_dir = image_dir
        self.suffix = suffix
        
        if stats is None:
            info = Stat4Data(image_dir=image_dir, suffix=suffix)
            stats = info._meanStdDev() # [2,3,1]
        
        self.mean_rgb = stats[0,:,0] # [3]
        self.std_rgb = stats[1,:,0] # [3]
        
        self.transform = transform
        
        self.rotate = rotate
        
        # parameters
        self.erase_shape = erase_shape
        self.erase_count = erase_count
        
        self.v_mirror = v_mirror
        self.h_mirror = h_mirror
        
        self.resize = resize
        
        self.crop = crop
        self.crop_shape = crop_shape
        
        self.data = []
        # save the items
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(self.suffix):
                self.data.append(item.split(self.suffix)[0])
        
    def __len__(self):
        return len(self.data)
    
    
    def _normalize(self,image):
        
        image =image.astype(np.float32)
        image -= self.mean_rgb
        
        input_ = image.copy()
        
        image[:,:,0] /= 3*self.std_rgb[0]
        image[:,:,1] /= 3*self.std_rgb[1]
        image[:,:,2] /= 3*self.std_rgb[2]
        
        index_ = image>1
        image[index_] = 1
        index_ = image<-1
        image[index_] = -1
        
        return input_,image
        
    
    def __getitem__(self,index):
        
        image = self.image_dir + "/" + self.data[index] + self.suffix
        
        # read in image and label
        if os.path.isfile(image):
            # cv2 load a image with the format (w,h,c) and a order of BGR
            # ---cv operator---
            image = cv2.imread(image) # [h,w,c], [B,G,R]
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # [B,G,R] to [R,G,B]
            #b,g,r = cv2.split(image)
            #image = cv2.merge([r,g,b])
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
        
        if np.random.random()>(1-self.v_mirror):
            image = np.flip(image,1).copy()
        
        if np.random.random()>(1-self.h_mirror):
            image = np.flip(image,0).copy()
        
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
        # mask [h,w,1] [0,1]
        mask = np.ones(shape=(image.shape[0], image.shape[1], 1), dtype = np.uint8)
        if self.erase_count == 1:
            offset =round((image.shape[0]-self.earse_shape[0])/2)
            end = offset + self.erase_shape[0]
        
            mask[offset:end,offset:end,:] = 0
        else:
            for c_ in range(self.erase_count):
                row = np.random.randint(0,image.shape[0] - self.erase_shape[0]+1)
                col = np.random.randint(0,image.shape[1] - self.erase_shape[1]+1)
                
                mask[row:row+self.erase_shape[0],col:col+self.erase_shape[1],:]=0
           
        input_,image = self._normalize(image)
        
        sample = {}
        sample["image"] = image
        sample["input_"] = input_
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
        input_ = sample["input_"]
        mask = sample["mask"]
        
        if self.transform is not None:
            print("Special process..")
            
            image = image.transpose(1,2,0)
            input_ = input_.transpose(1,2,0)
            mask = mask.transpose(1,2,0)
        
        input_ = input_ + self.mean_rgb
        input_ = input_.clip(0,255)
        
        image[:,:,0] *= 3*self.std_rgb[0]
        image[:,:,1] *= 3*self.std_rgb[1]
        image[:,:,2] *= 3*self.std_rgb[2]
        image += self.mean_rgb
        image = image.clip(0,255)
        
        fis,axs = plt.subplots(1,3)
        
        axs[0].imshow(input_.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(mask[:,:,0]*255,cmap="gray")
        axs[1].axis("off")
        axs[2].imshow(image.astype(np.uint8))
        axs[2].axis("off")
        
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return image,mask


class segmentation_data(data.Dataset):
    def __init__(self,image_dir,label_dir,
                 stats = None,
                 rotate = 0.5,
                 transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        if stats is None:
            info = Stat4Data(image_dir=image_dir, suffix=".tif")
            stats = info._meanStdDev() # [2,3,1]
        
        self.mean_rgb = stats[0,:,0] # [3]
        self.std_rgb = stats[1,:,0] # [3]
        
        self.rotate= rotate
        self.transform = transform
        
        self.data = []
        # save the items
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0])
        
    def __len__(self):
        return len(self.data)
    
    def _normalize(self,image):
        """
        it is directly copied from context_inpaint_data
        """
        image =image.astype(np.float32)
        image -= self.mean_rgb
        
        input_ = image.copy()
        
        image[:,:,0] /= 3*self.std_rgb[0]
        image[:,:,1] /= 3*self.std_rgb[1]
        image[:,:,2] /= 3*self.std_rgb[2]
        
        index_ = image>1
        image[index_] = 1
        index_ = image<-1
        image[index_] = -1
        
        return input_,image
    
    def __getitem__(self,index):
        image = self.image_dir + "/" + self.data[index] + ".tif"
        label = self.label_dir + "/" + self.data[index] + ".tif"
        # read in image and label
        if os.path.isfile(image) and os.path.isfile(label):
            # cv2 load a image with the format (w,h,c) and a order of BGR
            # ---cv operator---
            image = cv2.imread(image) # [h,w,c], [B,G,R]
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # [B,G,R] to [R,G,B]
            label = cv2.imread(label,flags=cv2.IMREAD_GRAYSCALE) # [h,w] [0,255]
            #label = label[:,:,np.newaxis] # [h,w,1]
        else:
            print("Coundn\'t find image:{} or its label!".format(image)) 
        
        # for rotation operator may change the value,
        # we process it in here.
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
            # image
            image = cv2.warpAffine(image,rot_mat,dsize=dsize,flags=cv2.INTER_LINEAR)
            # label
            label = cv2.warpAffine(label,rot_mat,dsize=dsize,flags=cv2.INTER_LINEAR)
            #print(label.shape)
            #print(label.dtype)
            #print(np.unique(label))
        
        input_,image = self._normalize(image) # np.float32 [h,w,c]
        label = label.astype(np.float32) 
        label = label[:,:,np.newaxis]*1./255 # np.float32 [h,w,1]
        
        sample = {}
        sample["image"] = input_
        sample["label"] = label
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def show_patch(self,index):
        
        sample  = self.__getitem__(index)
        image,label = sample["image"],sample["label"]
        
        if self.transform is not None:
            print("Special process..")
            image = image.transpose(1,2,0)
            label = label.transpose(1,2,0)
        
        # Here, image is actually the input_
        image = image + self.mean_rgb
        image = image.clip(0,255)
        
        label = label[:,:,0]
        label = label.clip(0,1)
        
        fis,axs = plt.subplots(1,2)
        
        axs[0].imshow(image.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(label.astype(np.uint8)*255,cmap="gray")
        axs[1].axis("off")

if __name__=="__main__":
    
    print("dataio.py testing...\n")
    
    
    data_dir = r"C:\Users\huijian\Downloads\repo\data\AerialImageDataset"

    image_dir = data_dir +"/" + "images"
    label_dir = data_dir + "/" + "gt"
    
    # pre process
    # info = Stat4Data(image_dir=image_dir,suffix=".tif")
    # stats = info._meanStdDev()
    
    
    # (1) context_inpaint_data
    # data_transform = None
    data_transform = torchvision.transforms.Compose([Nptranspose()])
    inpaint_dataset = context_inpaint_data(image_dir=image_dir,stats=AerialImageDataset_stats,
                                           erase_shape = [16,16],erase_count = 16,
                                           rotate = 0.5, 
                                           resize=0.5,
                                           crop = True, crop_shape = [128,128],
                                           transform=data_transform)
    inpaint_sample = inpaint_dataset[5]
    image = inpaint_sample["image"]
    mask = inpaint_sample["mask"]
    inpaint_dataset.show_patch(5)

    
    # 进行统计时，需要统计每个波段的平均值等，也需要统计类别的统计特性
    
    # (2) segmentation_data
    # data_transform = None
    data_transform = torchvision.transforms.Compose([RandomCrop(512),
                                                     H_Mirror(),V_Mirror(),
                                                     Nptranspose()])
    segmentation_dataset = segmentation_data(image_dir=image_dir,label_dir=label_dir,
                                             stats=AerialImageDataset_stats,
                                             transform=data_transform)
    segmentation_sample = segmentation_dataset[5]
    segmentation_dataset.show_patch(5)