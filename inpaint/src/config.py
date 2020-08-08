# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:14:35 2020

@author: huijian
"""

# mean RGB values of images
AerialImageDataset_mean_rgb = np.array([[103.60683725],[109.06976655],[100.39146181]])   
# standard deviation RGB values of images  
AerialImageDataset_std_rgb = np.array([[48.61960021],[44.44692765],[41.98457744]])   
AerialImageDataset_stats = np.array([AerialImageDataset_mean_rgb,
                                     AerialImageDataset_std_rgb])  