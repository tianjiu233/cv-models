# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:13:39 2020

@author: huijianpzh
"""
# torch libs
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# other official libs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# my libs
import metric
from loss.focalloss import FocalLoss
import loss.lovasz_losses as L

class Trainer(object):
    
    def __init__(self,
                 net,
                 cuda=False,
                 model_path="../checkpoint/"):
        self.net = net
        
        self.cuda = cuda
        self.model_path = model_path
        
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def _sample(self,sample):
        """
        For the limitation of dataset, the mask is required.
        
        The speical process will be done when sampling.
        Only one other process will be done in computing loss
        """
        
        image,label = sample["image"],sample["label"]
        
        image =image.to(torch.float32)
        # from [b,1,h,w] to [b,h,w], 
        # usually the official loss fcn requires the format of [N,d1,d2...]
        label = label.squeeze(1) 
        label = label.long()
        
        if self.cuda:
            image = image.to(self.device)
            label = label.to(self.device)

        return image,label,None
    
    def save_model(self,model_name="seg.pkl"):
        if self.cuda:
            self.net = self.net.cpu()
        torch.save(self.net,self.model_path+"/"+model_name)    
        
        if self.cuda:
            self.net = self.net.to(self.device)
        print("model saved!")
        return 
    
    def restore_model(self,model_name="seg.pkl"):
        self.net = torch.load(self.model_path+"/"+model_name)
        if self.cuda:
            self.net = self.net.to(self.device)
        print("model restored!")
        return        
    
    def _loss(self,pred,target,mask=None):
        # pred: (batch,cls_num,height,width) 
        # target: (batch,height,width) val:0->cls_num-1
        
        # ce loss
        ce_loss = self.ce_loss(pred,target)
        # Lovász loss
        l_loss = L.lovasz_softmax(probas=F.softmax(pred,dim=1),labels=target)
        loss = 0.5*ce_loss + 0.5*l_loss
            
        return loss
    
    def _compute_info(self,pred,target):
        """
        pred: [batch_num,cls_num,height,width] torch.tensor
        target: [batch_num,1,height,width] torch.tensor
        """
        # multi-class classification
        # the information should include oa,kappa,accuracy for every class, iou and jaccard
        
        # conf_matrix
        # torch.max(tensor,dim): return of max val and the index on the certain dim
        
        cls_num = pred.size(1)
        _,new_pred = torch.max(pred,1)
        new_pred = new_pred.detach().cpu().numpy().squeeze().astype(np.uint8)
        new_target = target.detach().cpu().numpy().squeeze().astype(np.uint8)
        
        conf_mat = metric.confusion_matrix(pred=new_pred.flatten(),
                                           target=new_target.flatten(),
                                           cls_num = cls_num)
        
        accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa = metric.evalue(conf_mat)
        
        return conf_mat,accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa
    
    def validate(self,val_data_loader):
        # change state
        self.net.eval()
        
        total_conf_mat = 0
        for i,sample in tqdm(enumerate(self.val_loader,0)):
            # sample
            image,label,mask = self._sample(sample)
            
            with torch.no_grad():
                pred = self.net(image)
                
            cls_num = pred.size(1)
            _,new_pred = torch.max(pred,1)
            new_pred = new_pred.detach().cpu().numpy().squeeze().astype(np.uint8)
            new_target = label.squeeze().cpu().numpy().squeeze().astype(np.uint8)
            
            conf_mat = metric.confusion_matrix(pred=new_pred.flatten(),
                                               target=new_target.flatten(),
                                               cls_num = cls_num)
            total_conf_mat += conf_mat
            # -----------------------
            #print("\n")
            #print(conf_mat)
            #print("\n")
            # -----------------------
        # compute the final evaluation
        accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa = metric.evalue(total_conf_mat)
        # change state back
        self.net.train()
        return accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa
    
    def train_model(self,train_loader,val_loader,
                    epochs=int(1e6),
                    loss_accu_interval=1,val_interval=16,
                    model_name="seg",
                    optim_mode="Adam"):
        
        if self.cuda:
            self.net = self.net.to(self.device)
        
        # prepare data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # prepare loss functions
        # cross entropy
        self.ce_loss = nn.CrossEntropyLoss(weight=None,reduction="mean")
        
        # prepare the optimizer and its strategies
        if optim_mode == "Adam":
            self.optimizer = optim.Adam(params=self.net.parameters(),lr=3e-4,betas=(0.5,0.99))
        elif optim_mode == "Adadelta":
            self.optimizer = optim.Adadelta(params=self.net.parameters(),lr=0.1,weight_decay=0.0001)
        else:
            self.optimizer = optim.SGD(params=self.net.parameters(),lr=1e-2)
        

        # optimizer and loss initial
        self.optimizer.zero_grad()
        loss = 0
        self.net.train()
        
        # train
        for e in tqdm(range(epochs)):
            for i,sample in enumerate(self.train_loader,0):
                # train
                image,label =self._sample(sample,mask=self.mask)
                pred = self.net(image)
                # compute loss
                loss =self._loss(pred=pred,target=label)
                loss = loss/loss_accu_interval
                loss.backward()
                
                if ((i+1)%loss_accu_interval==0) or ((i+1)==len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                
                if (i+1)%(val_interval*50) == 0:
                    # every epoch, print the info of the certain batch
                    conf_mat,accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa = self._compute_info(pred=pred, target=label)
                    print("Epoch-{} Iteration-{}: training-info".format(e+1,i+1)) 
                    print("Accuray:{:.5f} || Kappa:{:.5f}".format(accu,kappa))
                    print("MIoU:{:.5f} || FWIoU:{:.5f}".format(mean_iou,fw_iou))
                    with open("train-info.txt","a") as file_handle:
                        file_handle.write("Epoch-{} Iteration-{}: training-info".format(e+1,i+1))
                        file_handle.write('\n')
                        file_handle.write("Accuray:{:.5f} || Kappa:{:.5f}".format(accu,kappa))
                        file_handle.write('\n')
                        file_handle.write("MIoU:{:.5f} || FWIoU:{:.5f}".format(mean_iou,fw_iou))
                        file_handle.write('\n')
            
            if (e+1)%val_interval == 0:
                accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa = self.validate(val_data_loader = self.val_loader)
                
                print("Epoch-{}: validating-info".format(e+1)) 
                print("Accuray:{:.5f} || Kappa:{:.5f}".format(accu,kappa))
                print("MIoU:{:.5f} || FWIoU:{:.5f}".format(mean_iou,fw_iou))
                with open("train-info.txt","a") as file_handle:
                    file_handle.write("Epoch-{}: validating-info".format(e+1))
                    file_handle.write('\n')
                    file_handle.write("Accuray:{:.5f} || Kappa:{:.5f}".format(accu,kappa))
                    file_handle.write('\n')
                    file_handle.write("MIoU:{:.5f} || FWIoU:{:.5f}".format(mean_iou,fw_iou))
                    file_handle.write('\n')
                sava_model_name = model_name + "_" + str(e+1) + ".pkl"
                
                self.save_model(sava_model_name)
        return