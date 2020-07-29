# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:57:28 2020

@author: huijian
"""

import os
import numpy as np
from skimage import io

def PatchImage(image,file_name,new_dir,suffix,
               patch_size=[512,512],stride=[256,256]):
    """
    image: the image to segment
    file_name: the name for the image
    new_dir: the dir to save the patches
    patch_size,stride: the parameters used for segmentation
    """
    
    image = io.imread(image)
    image_shape = image.shape
    
    # the first 1 is for the initial point
    # the second 1 is for the last point
    row_loop = (image_shape[0]-patch_size[0])//stride[0] + 1 + 1
    col_loop = (image_shape[1]-patch_size[1])//stride[1] + 1 + 1
    
    # two loops
    left_coor = 0
    top_coor = 0
    for i in range(row_loop):
        # compute the top_coor
        if i < row_loop - 1:
            top_coor = 0+stride[0]*i
        else:
            top_coor = image_shape[0] - patch_size[0]
        for j in range(col_loop):
            # compute the col_loop
            if j<col_loop-1:
                left_coor = 0 + stride[1] * j
            else:
                left_coor = image_shape[1] - patch_size[1]
            
            # create patch
            if len(image_shape) == 3:
                patch = np.empty(shape=(patch_size[0],patch_size[1],image_shape[2]),
                                 dtype=image.dtype)
            else:
                patch = np.empty(shape=(patch_size[0],patch_size[1]),dtype=image.dtype)
            
            if len(image_shape) == 3:
                patch = image[top_coor:top_coor + patch_size[0],left_coor:left_coor+patch_size[1],:]
            else:
                patch = image[top_coor:top_coor + patch_size[0],left_coor:left_coor+patch_size[1]]
            
            patch_name = new_dir + "/" + file_name + "_" + str(i) + "_" + str(j) + suffix
            io.imsave(patch_name,patch)



def CreateDataset(OriginalDataDir,OriginalLabelDir,
                  NewDataDir,NewLabelDir,
                  suffix = ".tif",
                  patch_size=[512,512],stride=[256,256]):
    
    
    # this part codes are supposed to be modified depending on the situation.
    data = []
    files = os.listdir(OriginalDataDir)
    for item in files:
        if item.endswith(suffix):
            data.append(item.split(suffix)[0])
    
    # read in the images and labels then divide them
    # create data dir
    assert not os.path.exists(NewDataDir)
    os.makedirs(NewDataDir)
    
    for file in data:
        
        image = OriginalDataDir + "/" + file + suffix 
        PatchImage(image = image,
                   file_name = file ,
                   new_dir = NewDataDir,
                   suffix=suffix,patch_size=patch_size,stride=stride)
        
        
    # create corresponding label dir
    assert not os.path.exists(NewLabelDir)
    os.makedirs(NewLabelDir)
    for file in data:
        
        label = OriginalLabelDir + "/" + file + "_label" + suffix
        PatchImage(image = label,
                   file_name = file + "_label" ,
                   new_dir = NewLabelDir,
                   suffix=suffix,patch_size=patch_size,stride=stride)
    
    return
    

if __name__ == "__main__":
    PathDir =r"D:/repo/data/GID/"

    OriginalDataDir = PathDir + "Fine_land-cover_Classification_15classes/" + "image_RGB"
    OriginalLabelDir = PathDir + "Fine_land-cover_Classification_15classes/" + "label_15classes"
    
    NewDataDir = OriginalDataDir + "_Patch"
    NewLabelDir = OriginalLabelDir + "_Patch"
    
    CreateDataset(OriginalDataDir,OriginalLabelDir,NewDataDir,NewLabelDir,suffix=".tif",patch_size=[1024,1024],stride=[512,512])