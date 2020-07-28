# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:59:15 2018

@author: huijian
"""

import torch
import torchvision
import numpy as np
from cvppp_data import (Leafs,Resize,Rotate,V_Mirror,H_Mirror)
from cvppp_arch import Architecture
from trainer import Trainer

from PIL import Image
from skimage import io,color
import matplotlib.pyplot as plt

from myutils import COLOR_DICT

DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
"coral","gold","lightblue","lightgrey","purple",
"silver","navy","maroon","olive","orange")

if __name__=="__main__":
    
    # prepare data
    maximum=20
    root_dir = "../data/A1"
    composed = torchvision.transforms.Compose([Resize(),Rotate(),V_Mirror(),H_Mirror()])
    transform = composed
    train_data = Leafs(root_dir = root_dir, transform=transform, resize_shape=(256,256), maximum=maximum)
    test_data = Leafs(root_dir = root_dir, transform=transform, resize_shape=(256,256), maximum=maximum)

    # define the model
    file_path = "../model"
    cuda = torch.cuda.is_available()
    pixel_embedding_dim = 32
    if True:
        net = Architecture(n_classes=2, use_instance_seg=True, use_coords=False, pixel_embedding_dim=pixel_embedding_dim)
        trainer = Trainer(net=net,file_path=file_path,cuda=cuda)
    else:
        trainer = Trainer(net=None,file_path=file_path,cuda=cuda)
        trainer.restore_model(model_name=None)
    
    # training the model
    max_n_objects = maximum
    epochs=300
    train_bs=4
    test_bs=1
    if True:
        trainer.train_model(train_data=train_data,test_data=test_data,max_n_objects=max_n_objects,
                            epochs=epochs,train_bs=train_bs,test_bs=test_bs)
    
    # show a sample(for test)
    if True:
        idx = np.random.randint(len(test_data))
        sample = test_data[idx]
        image = sample["image"] # (channels,height,width) np
        label = sample["label"]
        tmp = label
        label = color.label2rgb(label,bg_label=0,colors=DEFAULT_COLORS)
        semantic_mask = sample["semantic_mask"][0,:,:]
        
        input_image = torch.tensor(image)
        
        pred_sem_seg, pred_instance_mask,pred_n_objects = trainer.instance_predict(image = input_image,mask=semantic_mask)
        annotation1 = color.label2rgb(pred_instance_mask,bg_label=0,colors=DEFAULT_COLORS)
        annotation1 = annotation1.astype(np.float32)
        
        pred_sem_seg, pred_instance_mask,pred_n_objects = trainer.instance_predict(image = input_image,mask=None)
        annotation2 = color.label2rgb(pred_instance_mask,bg_label=0,colors=DEFAULT_COLORS)
        annotation2 = annotation2.astype(np.float32)
        
        annotation1 = (annotation1*255).astype(np.uint8)
        annotation2 = (annotation2*255).astype(np.uint8)
        semantic_mask = color.label2rgb(semantic_mask,bg_label=0)
        pred_sem_seg = color.label2rgb(pred_sem_seg,bg_label=0)
        
        fig,ax = plt.subplots(2,3)
        
        image = (image.transpose(1,2,0) + 1)*(255*0.5)
        image = image.astype(np.uint8)
        ax[0][0].imshow(image)
        ax[0][0].set_title("Image")
        ax[0][1].imshow(semantic_mask)
        ax[0][1].set_title("semantic_seg_gt")
        ax[0][2].imshow(label)
        ax[0][2].set_title("instance_seg_gt")
        ax[1,0].imshow(pred_sem_seg)
        ax[1][0].set_title("pred_sem_seg")
        ax[1][1].imshow(annotation1)
        ax[1][1].set_title("instace_masked_gt")
        ax[1][2].imshow(annotation2)
        ax[1][2].set_title("instance_masked_pred")
        plt.show()
        
        
    
    