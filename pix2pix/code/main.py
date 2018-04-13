#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:14:20 2018

@author: huijian
"""

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from dataio import ABDataset,RandomCrop,H_Mirror,V_Mirror,Jitter,Nptranspose
from pix2pix import Generator, Discriminator, weights_init_normal
from trainer import Trainer

from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def get_result(image,label,pred,save=True,show=False):
    """
    both items are Variables and 4-dim.
    batch_size should be one.
    """
    image = image.squeeze().data.numpy()
    label = label.squeeze().data.numpy()
    pred = pred.squeeze().data.numpy()
    
    image = (image.transpose(1,2,0)+1)*(255*0.5)
    label = (label.transpose(1,2,0)+1)*(255*0.5)
    pred = (pred.transpose(1,2,0)+1)*(255*0.5)
    
    if show:
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(image.astype(np.uint8))
        ax[1].imshow(label.astype(np.uint8))
        ax[2].imshow(pred.astype(np.uint8))
        ax[0].set_title("Image")
        ax[1].set_title("Label")
        ax[2].set_title("Pred")
        
        plt.show()
    if save:
        io.imsave("../image.jpg",image.astype(np.uint8))
        io.imsave("../label.jpg",label.astype(np.uint8))
        io.imsave("../pred.jpg",pred.astype(np.uint8))
    return
    

def run_dataset(dataset,trainer,root_dir="../testdataset_result/"):
    for i in range(len(dataset)):
        sample = dataset[i]
        image = Variable(torch.FloatTensor(sample["image"]))
        image = image.unsqueeze(0)
        
        if torch.cuda.is_available():
            image = image.cuda()
        
        pred = trainer.predict(image)
        
        if torch.cuda.is_available():
            pred = pred.cpu()
        
        pred = pred.squeeze()
        pred = (pred.data.numpy().transpose(1,2,0)+1)*(255*0.5)
        pred = pred.astype(np.uint8)
        str_name = root_dir+"pred_"+str(i)+".jpg"
        io.imsave(str_name, pred)
        
        label = sample["label"]
        label = (label.transpose(1,2,0)+1)*(255*0.5)
        label = label.astype(np.uint8)
        str_name = root_dir+"label_"+str(i)+".jpg"
        io.imsave(str_name,label)

if __name__=="__main__":
    
    cuda = torch.cuda.is_available()
    
    root_dir = "../maps/"
    composed = transforms.Compose([RandomCrop(256),Jitter(),H_Mirror(),V_Mirror(),Nptranspose()])
    #composed = transforms.Compose([RandomCrop(256),Nptranspose()])
    
    train_dataset = ABDataset(root_dir = root_dir+"train/",transform = composed)
    val_dataset = ABDataset(root_dir=root_dir+"val/",transform=composed)
    
    
    file_path = "../model/"
    
    if False:
        trainer = Trainer(Generator=None, Discriminator=None, file_path = file_path)
        trainer.restore_model()
    else:
        disc = Discriminator()
        gen = Generator()
        if cuda:
            disc = disc.cuda()
            gen = gen.cuda()
        disc.apply(weights_init_normal)
        gen.apply(weights_init_normal)
        trainer = Trainer(Generator=gen, Discriminator=disc, file_path = file_path)
    
    if True:
        trainer.train_model(train_data=train_dataset,
                            test_data=val_dataset,
                            batch_size=1,epochs=200)
    if True:
        composed = transforms.Compose([RandomCrop(512),Nptranspose()])
        test_dataset = ABDataset(root_dir = root_dir+"test/",transform = composed)
        i=np.random.randint(0,len(test_dataset))
        sample = test_dataset[i]
        image = Variable(torch.FloatTensor(sample["image"]))
        image = image.unsqueeze(0)
        label = Variable(torch.FloatTensor(sample["label"]))
        label = label.unsqueeze(0)
        
        if cuda:
            image=image.cuda()
            label=label.cuda()
        
        pred = trainer.predict(image)
        
        if cuda:
            image = image.cpu()
            label = label.cpu()
            pred = pred.cpu()
        
        get_result(image=image,label=label,pred=pred,save=False,show=True)
    
    if False:
        composed = transforms.Compose([RandomCrop(512),Nptranspose()])
        dataset=ABDataset(root_dir = root_dir+"train/",transform = composed)
        run_dataset(dataset,trainer,root_dir="../testdataset_result/")
        
        
    
