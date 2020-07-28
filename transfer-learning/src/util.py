# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:57:28 2020

@author: huijian
"""

import os
from skimage import io

def PatchImage(image,file_name,new_dir,
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
        for j in range(col_loop)



def CreateValData(OriginalValDataDir,OriginalLabelDir,
                  NewValDataDir,NewValLabelDir):
    
    suffix = ".tif"
    # this part codes are supposed to be modified depending on the situation.
    data = []
    files = os.listdir(OriginalValDataDir)
    for item in files:
        if item.endswith(suffix):
            data.append(item.split(suffix)[0])
    
    # read in the images and labels then divide them
            
    return
    

if __name__ == "__main__":
    DataDir= "../../data/GID/Fine_land-cover"
    OriginalValDataDir = 