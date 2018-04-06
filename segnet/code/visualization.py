#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:38:10 2018

@author: huijian
"""

import numpy as np
import matplotlib.pyplot as plt

# define the function for showing the picture
def visualize_results(image, image_true, image_pred):
    """
    image: [height, width, channels] 0-255
    image_true:[height, width] 0-1
    image_pred:[height, width] 0-1
    
    true positive(green): y_true = 1, y_pred = 1
    false positive(red): y_true = 0, y_pred = 1
    false negative(blue): y_true = 1, y_pred = 0
    true negative(original): y_true = 0, y_pred = 0
    """
    image_original = image.copy()

    mask = ((image_pred + image_true)>0.5)
    image[mask,:] = 0
    
    # Green
    mask = ((image_true*image_pred)>0)
    TP = mask.astype(np.int).sum()
    print("TP:{TP}".format(TP=TP))
    image[mask,1] = 255.0
    
    # Red
    mask = (((image_true<0.5).astype(np.uint8) * (image_pred>0.5).astype(np.uint8))>0.5)
    FP = mask.astype(np.int).sum()
    print("FP:{FP}".format(FP=FP))
    image[mask,0] = 255.0

    # Blue
    mask = (((image_true>0.5).astype(np.uint8) * (image_pred<0.5).astype(np.uint8))>0.5)
    FN = mask.astype(np.int).sum()
    print("FN:{FN}".format(FN=FN))
    image[mask,2] = 255.0


    TN = np.int(image.shape[0]*image.shape[1] - TP - FP - FN)
    print("TN:{TN}".format(TN=TN))
    precesion = (TP*1.)/(TP+FP)
    recall = (TP*1.)/(TP+FN)
    accuracy = 1.0*(TP + TN)/(image.shape[0]*image.shape[1])
    print(image.shape)
    print("This test: accuracy: {accuracy}; precesion: {precesion}; recall: {recall}".format(
        accuracy= accuracy,precesion=precesion, recall = recall))

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image_original.astype(np.uint8))
    ax[1].imshow(image.astype(np.uint8))
    ax[0].set_title("Input")
    ax[1].set_title("Prediction")
    plt.show()

    return

