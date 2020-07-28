# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:21:30 2020

@author: huijianpzh
"""

import numpy as np
from skimage import transform

class Add_Mask(object):
    """
    add the mask, which mean data is available
    """
    def __call__(self,sample):
        image,label = sample["image"],sample["label"]
        mask = np.zeros(shape=label.shape)
        
        mask[label>0.5] = 1
        #print(mask.shape)
        sample = { "image":image,"label":label,"mask":mask
            }
        
        return sample

class ColorAug(object):
    """
    Input Color Augmentation
    """
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        
        if np.random.random()<(1-self.p):
            return sample
        
        image,label = sample["image"],sample["label"]
        
        n_ch = image.shape[2]
        contrast_adj = 0.05
        bright_adj = 0.05
        
        ch_mean = np.mean(image,axis=(0,1),keepdims=True).astype(np.float32)
        
        contrast_mul = np.random.uniform(1 - contrast_adj, 1 + contrast_adj, (1, 1, n_ch)).astype(np.float32)   
        bright_mul = np.random.uniform(1 - bright_adj, 1 + bright_adj, (1, 1, n_ch)).astype(np.float32)
        
        # if contrast_mul = 1 & bright_mul = 1, then image = image
        image = (image - ch_mean)*contrast_mul + ch_mean * bright_mul
        
        # clip the value
        image = image.clip(min=0,max=1)
        
        
        return {'image':image,'label':label}

class Nptranspose(object):
    def __call__(self,sample):
        image = sample["image"]
        label = sample["label"]
        
        image = image.transpose(2,0,1)
        label = label.transpose(2,0,1)
        
        # normalize the image
        # image = image*1.0/255 
        
        sample["image"] = image
        sample["label"] = label
        
        return sample

class Rotation(object):
    def __init__(self,angle=90):
        self.angle = angle
    def __call__(self,sample):
        image,label= sample["image"],sample["label"]
        ids = np.around(360/self.angle)
        multi = np.random.randint(0,ids)
        if multi>0.001:
            #print("do rotation")
            # transform.rotate will change the range of the value
            image = transform.rotate(image,self.angle*multi).astype(np.float32)
            label = transform.rotate(label,self.angle*multi).astype(np.float32)
            sample["image"] = image
            sample["label"] = label
            #print(np.unique(label))
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
        
class StdCrop(object):
    """Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired ouput size, if int, square crop 
        is made.
    """
    def __init__(self,output_size,top=0,left=0):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
        self.top = top
        self.left = left
    def __call__(self,sample):
        image,label = sample["image"], sample["label"]
        h,w = image.shape[0:2]
        new_h, new_w = self.output_size
        
        top = self.top
        left = self.left
        
        image = image[top:top+new_h, left:left+new_w,:]
        label = label[top:top+new_h, left:left+new_w,:]
        
        return {"image":image,"label":label}