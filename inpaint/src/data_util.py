# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:10:38 2020

@author: huijianpzh
"""
import cv2
import os
import numpy as np
from tqdm import tqdm

class Stat4Data(object):
    """
    Do statistics for the image dataset, which will help get statitics info of the dataset. 
    """
    def __init__(self,image_dir,suffix):
        self.image_dir = image_dir
        self.suffix = suffix
        
        self.data = []
        # save the items
        files = os.listdir(self.image_dir)
        for item in files:
            if item.endswith(".tif"):
                self.data.append(item.split(".tif")[0])
        
    def _meanStdDev(self):
        
        stats = []
        for item in tqdm(self.data):
            image = self.image_dir + "/" + item + self.suffix
            # read in image
            image = cv2.imread(image) # [h,w,c], [B,G,R]
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #b,g,r = cv2.split(image)
            #image = cv2.merge([r,g,b])
            mean,std = cv2.meanStdDev(image)
            stats.append([mean,std])

        # stats [N,2,3,1]
        stats = np.array(stats)
        stats = np.mean(stats,axis=0)
        # stats [2,3,1]
        return stats

class Nptranspose(object):
    def __call__(self,sample):
        
        for item in sample.keys():
            mat_data = sample[item]
            mat_data = mat_data.transpose(2,0,1)
            sample[item] = mat_data
            
        return sample
    

class H_Mirror(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            new_image = np.flip(image,0).copy()
            new_label = np.flip(label,0).copy()
            
            return {'image':new_image, 'label':new_label}
        else:
            return {'image':image,'label':label}

class V_Mirror(object):
    def __init__(self,p = 0.5):
        self.p = p
    def __call__(self,sample):
        image, label= sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            new_image = np.flip(image,1).copy()
            new_label = np.flip(label,1).copy()

            return {'image':new_image, 'label':new_label}
        else:
            return {'image':image,'label':label}
        
class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size=(output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
    def __call__(self,sample):
        image,label = sample["image"], sample["label"]
        h,w = image.shape[0:2]
        
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        
        image = image[top:top+new_h, left:left+new_w,:]
        label = label[top:top+new_h, left:left+new_w,:]

        return {"image":image,"label":label}