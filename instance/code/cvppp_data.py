# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:06:04 2018

@author: huijian
"""

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchvision

"""
0:background
1:target
"""

def post_process_label(label,maximum):
    instance_values = set(np.unique(label)).difference([0]) # exclue value "0"
    instance_values = list(instance_values)
    n_instances = len(instance_values)
    height,width = label.shape[0:2]
    instance_mask = np.zeros(
            (height,width,maximum),dtype=np.uint8
            )
    for idx in range(maximum):
        if idx < n_instances:
            _mask = np.zeros((height,width),dtype=np.uint8)
            value = instance_values[idx]
            _mask[label==value]=1
            instance_mask[:,:,idx] = _mask
        else:
            _mask = np.zeros((height,width),dtype=np.uint8)
            instance_mask[:,:,idx] = _mask
        
    semantic_mask = instance_mask.sum(2)
    semantic_mask[semantic_mask!=0]=1
    semantic_mask = semantic_mask.astype(np.uint8)
    return instance_mask,n_instances,semantic_mask


class Resize(object):
    def __init__(self,output_shape=(256,256)):
        self.output_shape = output_shape
    def __call__(self,sample):
        image,label = sample["image"],sample["label"]
        image = image.resize(self.output_shape,Image.NEAREST)
        label = label.resize(self.output_shape,Image.NEAREST)
        sample["image"],sample["label"] = image,label
        return sample

class Rotate(object):
    def __init__(self,angle=90):
        self.angle = angle
    def __call__(self,sample):
        ids = np.around(360/self.angle)
        multi = np.random.randint(0,ids)
        if multi>0.01:
            image,label = sample["image"],sample["label"]
            image = image.rotate(multi*self.angle)
            label = label.rotate(multi*self.angle)
            sample["image"],sample["label"] = image,label
        return sample

class H_Mirror(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        if np.random.random()<self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
            sample["image"],sample["label"] = image,label
        return sample

class V_Mirror(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        if np.random.random()<self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            sample["image"],sample["label"] = image,label
        return sample
        

class Leafs(Dataset):
    def __init__(self,root_dir,transform=None,resize_shape=None,maximum=20):
        
        self.root_dir = root_dir
        self.transform = transform
        self.resize_shape = resize_shape
        self.maximum = maximum
        
        self.data = []
        
        # save the items
        files = os.listdir(self.root_dir)
        for item in files:
            if "label" in item:
                self.data.append(item.split("_label")[0])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        
        image = self.root_dir + "/" + self.data[index] + "_rgb.png"
        label = self.root_dir + "/" + self.data[index] + "_label.png"
        #print(image)
        image = Image.open(image) # (height,width,channel) 0~255
        label = Image.open(label) 
        
        sample = {}
        sample["image"] = image
        sample["label"] = label
        
        if self.transform:
            sample = self.transform(sample)
        
        image = sample["image"]
        label = sample["label"]
        
        image = np.array(image)[:,:,:3] # 0~255, 3channels
        image = image.astype(np.float32)/(255*0.5) - 1 # -1~1
        label = np.array(label) # (height,width)
        
        instance_mask,n_objects,semantic_mask = post_process_label(label=label,maximum=self.maximum)
        
        sample = {}
        sample["image"] = image
        sample["n_objects"] = np.array([n_objects],dtype = np.int) # when .__sample it will turn to be torch.int32
        sample["instance_mask"] = instance_mask # (height,width,max_n_objects),maximum 0,1
        sample["semantic_mask"] = semantic_mask # (height,width) 0,1
        sample["label"] = label
        
        if True:
            image = image.transpose(2,0,1)
            height,width = semantic_mask.shape
            semantic_mask = semantic_mask.reshape(1,height,width)
            instance_mask = instance_mask.transpose(2,0,1)
            sample["image"] = image #(channel,height,width)
            sample["semantic_mask"] = semantic_mask #(1,height,width)
            sample["instance_mask"] = instance_mask #(max_n_objects,height,width)
        return sample

def show_sample(sample):
    semantic_mask = sample["semantic_mask"]
    instance_mask = sample["instance_mask"]
    image=  sample["image"]
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(((image+1)*(255*0.5)).astype(np.uint8))
    ax[1].imshow(semantic_mask.astype(np.uint8))
    ax[2].imshow(instance_mask[:,:,1].astype(np.uint8))

if __name__=="__main__":
    root_dir = "../data/A1"
    composed = torchvision.transforms.Compose([Resize(),Rotate(),V_Mirror(),H_Mirror()])
    transform = composed
    leafs = Leafs(root_dir = root_dir, transform=transform, resize_shape=(256,256), maximum=20)
    sample = leafs[0]
    image = sample["image"] #(channel,height,width)
    instance_mask = sample["instance_mask"] #(height,width,max_n_objects) 
    semantic_mask = sample["semantic_mask"] #(1,height,width)
    
    sample["image"] = image.transpose(1,2,0)
    sample["semantic_mask"] = semantic_mask.reshape(semantic_mask.shape[1],semantic_mask.shape[2])
    sample["instance_mask"] = instance_mask.transpose(1,2,0)
    show_sample(sample) 
    
    # using DataLoader
    data_loader = DataLoader(dataset=leafs,batch_size=4,shuffle=False)
    data_loader = enumerate(data_loader)
    sample = next(data_loader)[1]
    # sample["n_objects"] shape-> [4,1]
    print("image")
    print(sample["image"].size(), sample["image"].dtype)
    print("semantic_mask")
    print(sample["semantic_mask"].size(),sample["semantic_mask"].dtype)
    print("instance_mask")
    print(sample["instance_mask"].size(),sample["instance_mask"].dtype)
    print("n_objects")
    print(sample["n_objects"].size(),sample["n_objects"].dtype)
    
    
    