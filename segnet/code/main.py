#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:08:49 2018

@author: huijian
"""

import torch
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms

# my libs
from dataio import BuildingDataset
from dataio import RandomCrop

from model import SegNet, weights_normal
from trainer import Trainer

from visualization import visualize_results

if __name__ == "__main__":
    
    train_dir = "/home/huijian/exps/Data/building_UT/train/"
    test_dir = "/home/huijian/exps/Data/building_UT/test/"
    
    composed = transforms.Compose([RandomCrop(256)])
    
    file_path = "/home/huijian/exps/segnet/model/"
    train_data = BuildingDataset(root_dir = train_dir, transform = composed)
    test_data = BuildingDataset(root_dir = test_dir, transform = composed)
    
    train_loader = utils.data.DataLoader(train_data,batch_size=4,shuffle=True)
    test_loader = utils.data.DataLoader(test_data,batch_size=4,shuffle=True)
    
    # building the net
    # [64,128,256,512,512]
    segnet = SegNet(input_dim=3, output_dim=1, features=[64,96,128,256,256])
    segnet.apply(weights_normal)
    
    print(segnet)
    
    trainer = Trainer(net = segnet, file_path = file_path)
    
    # restore the model
    if True:
        trainer.restore_model()
    
    # begin training
    if False:
        print("begin training!")
        trainer.train_model(train_loader = train_loader, test_loader = test_loader, epoch=100)
    
    # for show re-define the test_data
    test_data = BuildingDataset(root_dir=test_dir, transform = transforms.Compose([RandomCrop(768)]))
    # show result:
    if True:
        # redifine test
        sample=test_data[1]
        image_pred = trainer.predict(Variable(torch.FloatTensor(sample["image"])).unsqueeze(0))
        image_pred = image_pred.squeeze().data.numpy()
        image = (sample["image"].transpose(1,2,0)+1)*(255*0.5)
        label = sample["label"].transpose(1,2,0)[:,:,0]
        visualize_results(image, label, image_pred)