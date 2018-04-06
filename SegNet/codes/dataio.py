#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:59:49 2018

@author: huijian
"""
import torchvision.transforms as transforms
import torch.utils as utils 
from skimage import io

import numpy as np
import os


class RandomCrop(object):
    """Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired ouput size, if int, square crop 
        is made.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        h,w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        
        image = image[top:top+new_h, left:left+new_w]
        label = label[top:top+new_h,left:left+new_w]
        return {'image':image, 'label':label}


class OrchardDataset(utils.data.Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.train_data = []
        files = os.listdir(self.root_dir)
        for item in files:
            if item.endswith(".png"):
                self.train_data.append(item.split(".png")[0])

        
        
    def __len__(self):
        return len(self.train_data)
    def __getitem__(self, index):
        
        prefix = self.root_dir+"/"+self.train_data[index]
        img_name = prefix+".png"
        image = io.imread(img_name)
        # in this case, only three channels are used
        image =image[:,:,0:3]
        
        label_name = prefix+"_json/"+ "label.png"
        label = io.imread(label_name)
        
        sample = {}
        sample['image'] = image.astype(np.float32)/(255*0.5) - 1
        sample['label'] = label.astype(np.float32)
        
        if self.transform:
            sample = self.transform(sample)
        
        sample["image"] = sample["image"].transpose(2,0,1)
        label = sample["label"]
        sample["label"] = label.reshape(1,label.shape[0],label.shape[1])
        
        return sample

if __name__ == "__main__":
    print("dataio.py testing...")
    composed = transforms.Compose([RandomCrop(512)])
    train_dataset = OrchardDataset(root_dir = "/home/huijian/exps/Data/orchard_sample/train/",transform = composed)
    #test_dataset = OrchardDataset(root_dir="/home/huijian/exps/Data/orchard_sample/test/",transform=composed)
    
    sample = train_dataset[1]
    image = sample["image"]
    label = sample["label"]
    
    train_loader = utils.data.DataLoader(train_dataset,
                                           batch_size=4,
                                           shuffle=True)
    
    