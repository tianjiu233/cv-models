#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:21:55 2018

@author: huijian
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

class Trainer(object):
    def __init__(self, net, file_path):
        self.file_path = file_path
        self.net = net
    
    def train_model(self, train_loader, test_loader, epoch=300):
        
        self.net.train()
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(),lr=1e-3,betas=(0.9,0.99))
        
        for i in range(epoch):
            for j,sample in enumerate(train_loader,0):
                
                optimizer.zero_grad()
                
                image = Variable(sample["image"],requires_grad=False)
                label = Variable(sample["label"],requires_grad=False)
                
                prediction = self.net(image)
                
                loss = criterion(prediction,label)
    
                loss.backward()
                optimizer.step()
                
            # every epoch, compute the accuracy
            prediction = prediction.view(-1)
            label = label.view(-1)
            #balance = Variable(torch.FloatTensor([0.5]*len(prediction)))
            balance = 0.5
            prediction = torch.ge(prediction,balance).type(torch.FloatTensor)
            accuracy = torch.eq(label,prediction).type(torch.FloatTensor)
            accuracy = torch.div(accuracy.sum(),len(label))
                
            print("Train-Epoch:{}/{}, loss:{}, accuracy:{}".format(i+1,epoch,loss.data.numpy()[0], accuracy.data.numpy()[0]))
            self.save_model()
            
            if (i+1)%10 == 0:
                self.net.eval()
                loss = 0
                accuracy = 0
                for j,sample in enumerate(test_loader,0):
                    
                    image = Variable(sample["image"],requires_grad=False)
                    label = Variable(sample["label"],requires_grad=False)
                    prediction = self.net(image)
                    loss = loss + criterion(prediction,label)
                    
                    prediction = prediction.view(-1)
                    label = label.view(-1)
                    balance = 0.5
                    prediction = torch.ge(prediction,balance).type(torch.FloatTensor)
                    tmp_accuracy = torch.eq(label,prediction).type(torch.FloatTensor)
                    tmp_accuracy = torch.div(tmp_accuracy.sum(),len(label))
                    accuracy = accuracy + tmp_accuracy
                
                accuracy = torch.div(accuracy, len(test_loader))
                loss = torch.div(loss, len(test_loader))
                print("Test-Epoch:{}/{}, loss:{}, accuracy:{}".format(i+1,epoch,loss.data.numpy()[0], accuracy.data.numpy()[0]))
                
    
    def save_model(self):
        torch.save(self.net, self.file_path + "unet_model.pkl")
    
    def restore_model(self):
        print("restore the model...")
        self.net = torch.load(self.file_path + "unet_model.pkl")
        
    def predict(self, image):
        self.net.eval()
        prediction = self.net(image)
        return prediction
