#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:38:36 2018

@author: huijian
"""

import numpy as np
import cv2
from skimage import io
if __name__=="__main__":
    
    # numerical 
    demo_mat = [[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]]
    demo_mat = np.array(demo_mat,dtype=np.uint8)
    dst_demo = cv2.distanceTransform(src=1-demo_mat,distanceType=cv2.DIST_L2,maskSize=cv2.DIST_MASK_PRECISE)
    
    # picture
    bw = np.zeros((200,200),dtype=np.uint8)
    bw[50,50]=1
    bw[50,150]=1
    bw[150,100]=1
    # Euclidean
    dst1 = cv2.distanceTransform(src=1-bw,distanceType=cv2.DIST_L2,maskSize=5)
    # city block(L1)
    dst2 = cv2.distanceTransform(src=1-bw,distanceType=cv2.DIST_L1,maskSize=5)
    # chessboard
    dst3 = cv2.distanceTransform(src=1-bw,distanceType=cv2.DIST_C,maskSize=5)
    
    mat = dst3
    v_max = mat.max()
    v_min = mat.min()
    new_mat = ((mat-v_min)/(v_max-v_min)*255).astype(np.uint8)
    io.imsave("chessboard.jpg",new_mat)