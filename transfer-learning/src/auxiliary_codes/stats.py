# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:47:39 2020

@author: huijianpzh
"""

import numpy as np
import os

from skimage import io

# data related
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

GF_lut = build_color2label_lut(colormap)

def color2label(color_img,lut=GF_lut):
    """
    color_img: [height,weight,3]
    lut: 1-d np.array
    """
    label = color_img.astype("int32")
    indices = label[:,:,0]*65536+label[:,:,1]*256 + label[:,:,2]
    
    return lut[indices]

def _process(label):
    
    label = label[:,:,0:3]
    label = color2label(color_img = label)
    
    return label


def median_frequency_balancing(label_files,cls_num):
    """Perform median frequency balancing on the image files, given by the formula
    
    f = Median_frequency_c / total_freq_C
        
    """
    
    label2frequency_dict = {}
    for i in range(cls_num):
        label2frequency_dict[i] = []
    
    for idx in range(len(label_files)):
        
        label = io.imread(label_files[idx])
        label = _process(label)
        
        for cls_idx in range(cls_num):
            cls_mask = np.equal(label,cls_idx)
            cls_mask = cls_mask.astype(np.float32)
            cls_frequency = np.sum(cls_mask)
            
            if cls_frequency != 0:
                label2frequency_dict[cls_idx].append(cls_frequency)
    
    # get pixels per cls
    pixels_per_cls = {}
    for cls_idx,pixels_list in label2frequency_dict.items():
        pixels_num = sum(pixels_list)
        pixels_per_cls[cls_idx] = pixels_num
    
    # get the cls_weights (mediam frequency weights)
    cls_weights = []
    # get the total pixels
    total_pixels = 0
    for pixels in label2frequency_dict.values():
        total_pixels += sum(pixels)
    
    for i,j in label2frequency_dict.items():
        j = sorted(j) # to get the median, we got to sort the frequencies
        
        median_frequency = np.median(j)/sum(j)
        total_frequency = sum(j) / total_pixels
        
        median_frequency_balanced = median_frequency / total_frequency
        cls_weights.append(median_frequency_balanced)
        
    return cls_weights,pixels_per_cls



if __name__ == "__main__":
    
    label_path = r"D:\repo\data\GF\data"
    
    label_files= []
    files = os.listdir(label_path)
    for item in files:
        if "_gt" in item:
            label_files.append(label_path+"/"+item)
                               
    cls_weights,pixels_per_cls = median_frequency_balancing(label_files,cls_num=9)
        
    