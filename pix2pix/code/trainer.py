#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:21:27 2018

@author: huijian
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision.transforms as transforms

import numpy as np

from torch.autograd import Variable

from dataio import ABDataset,RandomCrop
from skimage import io


class Trainer(object):
    def __init__(self, Generator, Discriminator,file_path):
        self.disc = Discriminator
        self.gen = Generator
        self.file_path = file_path
    def train_model(self,train_data,test_data,batch_size=4,epochs=100):
        
        cuda = torch.cuda.is_available()
        # prepare data
        train = utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
        #test = utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
        
        l1 = nn.L1Loss()
        bce = nn.BCELoss()
        
        D_optim = optim.Adam(self.disc.parameters(),lr=2e-4,betas=(0.9,0.999))
        G_optim = optim.Adam(self.gen.parameters(),lr=2e-4,betas=(0.9,0.999))
        
        
        for e_id in range(epochs):
            self.disc.train()
            self.gen.train()
            for (idx,sample) in enumerate(train,0):

                image = Variable(sample["image"])
                label = Variable(sample["label"])
                
                if cuda:
                    image = image.cuda()
                    label = label.cuda()
                
                
                # updates D:
                D_optim.zero_grad()
                #real data
                pred_real = self.disc(torch.cat([image,label],1))
                real = Variable(torch.ones(pred_real.shape),requires_grad=False)
                if torch.cuda.is_available():
                    real = real.cuda()
                real_loss = bce(pred_real,real)
                
                # fake part:
                fake_label = self.gen(image)
                pred_fake = self.disc(torch.cat([image,fake_label.detach()],1))
                fake = Variable(torch.ones(pred_fake.shape),requires_grad=False)
                if torch.cuda.is_available():
                    fake = fake.cuda()
                fake.data.fill_(0)
                fake_loss = bce(pred_fake,fake)
                
                D_loss = (fake_loss + real_loss)*0.5
                D_loss.backward()
                D_optim.step()
                
                # update G
                G_optim.zero_grad()
                pred = self.disc(torch.cat([image,fake_label],1))
                gan_loss = bce(pred,real)
                l1_loss = l1(fake_label,label)
                G_loss = gan_loss + 100*l1_loss
                G_loss.backward()
                G_optim.step()
                
                # print the training process
                
                if ((idx+1)%100==0):
                    if torch.cuda.is_available():
                        gan_loss = gan_loss.cpu()
                        l1_loss = l1_loss.cpu()
                        D_loss = D_loss.cpu()
                    print("Train:Epoch/idx:{}/{} G_loss:{:.5f}, L1_loss:{:.5f}, D_loss:{:.5f}".format(
                            e_id+1,idx+1,gan_loss.data.numpy()[0], l1_loss.data.numpy()[0], D_loss.data.numpy()[0]))
                """
                if ((idx+1)%100==0):
                    print("Train:Epoch/idx:{}/{} G_loss:{:.5f}, D_loss:{:.5f}".format(
                            e_id+1,idx+1,gan_loss.data.numpy()[0], D_loss.data.numpy()[0]))
                """
            if (e_id+1)%1==0:
                self.__save_tmp(image=image,label=label,e_id=e_id)
                self.__save_model()

                    
    def __save_model(self):
        if torch.cuda.is_available():
            self.disc = self.disc.cpu()
            self.gen = self.gen.cpu()
        torch.save(self.disc,self.file_path+"unet_discriminator.pkl")
        torch.save(self.gen,self.file_path+"generator.pkl")
        if torch.cuda.is_available():
            self.disc = self.disc.cuda()
            self.gen = self.gen.cuda()
        print("model saved!")
        return
    
    def __save_tmp(self,image,label,e_id):
        
        result = self.gen(image).squeeze()
        if torch.cuda.is_available():
            result = result.cpu()
        result = (result.data.numpy().transpose(1,2,0)+1)*(255*0.5)
        result = result.astype(np.uint8)
        result_name = "../result/result_epoch_"+str(e_id)+".jpg"
        io.imsave(result_name, result)
        
        label = label.squeeze()
        if torch.cuda.is_available():
            label = label.cpu()
        label = (label.data.numpy().transpose(1,2,0)+1)*(255*0.5)
        label = label.astype(np.uint8)
        label_name = "../result/label_epoch_"+str(e_id)+".jpg"
        io.imsave(label_name, label)
        
        
    
    def restore_model(self):
        disc = torch.load(self.file_path+"unet_discriminator.pkl")
        gen = torch.load(self.file_path+"generator.pkl")
        if torch.cuda.is_available():
            disc = disc.cuda()
            gen = gen.cuda()
        self.disc = disc
        self.gen = gen
        print("model restored!")
    
    def predict(self,image):
        self.gen.eval()
        prediction = self.gen(image)
        return prediction
        

if __name__=="__main__":
    
    root_dir = "../maps/"
    composed = transforms.Compose([RandomCrop(512)])
    train_dataset = ABDataset(root_dir = root_dir+"train/",
                               transform = composed)
    test_dataset = ABDataset(root_dir = root_dir+"test/",
                                transform = composed)
