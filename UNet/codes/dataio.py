#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:06:02 2018

@author: huijian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:59:49 2018

@author: huijian
"""
import torchvision.transforms as transforms
import torch.utils as utils 
from skimage import io

import matplotlib.pyplot as plt
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
        h,w = image.shape[1:3]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        
        image = image[:,top:top+new_h, left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]
        return {'image':image, 'label':label}


class BuildingDataset(utils.data.Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.train_data = []
        files = os.listdir(os.path.join(self.root_dir,"input"))
        for item in files:
            if item.endswith(".tiff"):
                self.train_data.append(item.split(".tiff")[0])
        
        
    def __len__(self):
        return len(self.train_data)
    def __getitem__(self, index):
        
        prefix = self.root_dir
        img_name = prefix+"input/"+self.train_data[index]+".tiff"
        image = io.imread(img_name)
        image = image.astype(np.float32)/(255*0.5) - 1
        
        label_name = prefix+"target/"+ self.train_data[index]+".tif"
        label = io.imread(label_name)[:,:,:1]
        label = label.astype(np.float32)/255.0
        
        sample = {}
        sample['image'] = image.transpose(2,0,1)
        sample['label'] = label.transpose(2,0,1)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

if __name__ == "__main__":
    print("dataio.py testing...")
    composed = transforms.Compose([RandomCrop(256)])
    building_dataset = BuildingDataset(root_dir = "/home/huijian/exps/Data/building_UT/train/",transform = composed)
    
    sample = building_dataset[1]
    # to show, change the axis
    image = (sample["image"].transpose(1,2,0) + 1)*(255*0.5)
    label = sample["label"].transpose(1,2,0)[:,:,0]
    
    if True:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(image.astype(np.uint8))
        ax[1].imshow(label.astype(np.uint8))
        ax[0].set_title("Image")
        ax[1].set_title("Label")
        plt.show()
    
    
    building_loader = utils.data.DataLoader(building_dataset,
                                           batch_size=4,
                                           shuffle=True)
    
    