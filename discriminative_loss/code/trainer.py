# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:37:33 2018

@author: huijian
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from discriminative import DiscriminativeLoss
#from new_discriminative import DiscriminativeLoss
import numpy as np
from sklearn.cluster import SpectralClustering

class Trainer(object):
    def __init__(self,net,file_path="../model",cuda=False,max_n_objects=20):
        self.net = net
        self.file_path = file_path
        self.cuda = cuda
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = None
        
        self.max_n_objects = max_n_objects # it is an attribute of the dataset and it can be corrected in .train_model fcn
        
    def __sample(self,sample):
        
        """
        get the data and change the data to the right format
        """
        image = sample["image"].to(torch.float32) # (bs,channels,height,width) torch.float32
        n_objects = sample["n_objects"] #(bs,1)
        semantic_mask = sample["semantic_mask"].long() # (bs,height,width) torch.int64
        instance_mask = sample["instance_mask"].long() # (bs,height,width,max_n_objects) torch.int64
        if self.cuda:
            image = image.to(self.device)
            n_objects = n_objects.to(self.device)
            semantic_mask = semantic_mask.to(self.device)
            instance_mask = instance_mask.to(self.device)
        return image,n_objects,instance_mask,semantic_mask
    
    def __compute_mul_accu(self,pred,gt):
        """
        pred: (bs,n_class,height,width)
        """
        (bs,_,height,width) = pred.size()
        n_pixels = bs*height*width
        
        _,pred = torch.max(pred,dim=1) # pred:(bs,n_class,h,w)->(bs,h,w)
        pred = pred.view(-1)
        
        gt = gt.view(-1) # gt: (bs,1,h,w) ->(bs*w*h)

        accu = torch.eq(pred,gt).to(torch.float32)
        accu = torch.sum(accu)
        accu = torch.div(accu,n_pixels)
        
        if self.cuda:
            accu = accu.cpu()
        
        accu = accu.detach().numpy() # np
        return accu
    
    def train_step(self,sample,iters,epoch):
        
        # clear the optimizer
        self.optimizer.zero_grad()
        
        image,n_objects,instance_mask,semantic_mask = self.__sample(sample)
        """
        image:(bs,channel,height,width) torch.float32
        n_objects: (bs,1) torch.int ?
        instance_mask:(bs,max_n_objects,height,width) torch.int64
        sematic_mask:(bs,n_classes,height,width) torch.int64
        """
        pred_sem_seg, pred_ins_seg, pred_n_objects_normalized = self.net(image)
        """
        pred_sem_seg:(bs,n_classes,height,width) torch.float32
        pred_ins_Seg:(bs,n_filters,height,width) torch.float32
        pred_n_objects_normalized:(bs,1) torch.float32
        """
        
        # CrossEntropy for semantic_segmantation (checked)
        n_classes = pred_sem_seg.size(1)
        ce_loss =  self.criterion_ce(pred_sem_seg.permute(0,2,3,1).contiguous().view(-1,n_classes),semantic_mask.view(-1))
        # Discriminative Loss, the n_objects and max_n_objects used here are true values
        disc_loss = self.criterion_discriminative(input=pred_ins_seg,target=instance_mask,n_objects=n_objects,max_n_objects=self.max_n_objects)
        # MSE Loss (checked)
        n_objects = n_objects.to(torch.float32)
        n_objects_normalized = torch.div(n_objects,self.max_n_objects)
        mse_loss = self.criterion_mse(input=pred_n_objects_normalized,target=n_objects_normalized)
        
        loss = mse_loss + ce_loss + disc_loss
        #loss = ce_loss
        loss.backward()
        
        if self.clip_grad_norm!=0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=self.clip_grad_norm,norm_type=2)
        self.optimizer.step()
        
        if (iters+1)%10==0:
            if self.cuda:
                ce_loss = ce_loss.cpu()
                #disc_loss = disc_loss.cpu()
                #mse_loss = mse_loss.cpu()
                
            ce_loss = float(ce_loss.detach().numpy())
            disc_loss = float(disc_loss.detach().numpy())
            mse_loss = float(mse_loss.detach().numpy())
            
            accu = self.__compute_mul_accu(pred=pred_sem_seg,gt=semantic_mask)
            """
            print("(Train)Epoch:{}/Iters:{} - Accu:{:.5}, CE_loss:{:.5}".format(
                    epoch+1,iters+1,accu,ce_loss))
            """
            print("(Train)Epoch:{}/Iters:{} - Accu:{:.5}, CE_loss:{:.5}, MSE_loss:{:.5}, disc_loss:{:.5}".format(
                    epoch+1,iters+1,accu,ce_loss,mse_loss,disc_loss))
            
        
    
    def validate_step(self):
        return
    
    def train_model(self,train_data,test_data,max_n_objects,epochs=300,train_bs=4,test_bs=1):
        
        if self.cuda:
            self.net = self.net.to(self.device)
        
        self.max_n_objects = max_n_objects # it is an attribute of the dataset.
        self.train_loader = DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True)
        self.test_loader = DataLoader(dataset=test_data,batch_size=test_bs,shuffle=False)
        
        # define the criterion
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        # model setting
        self.criterion_discriminative = DiscriminativeLoss(delta_var=0.5,delta_dist=1.5,norm=2,
                                                            cuda=self.cuda,device=self.device)
        
        # define the optimizer
        self.optimizer = optim.Adam(params=self.net.parameters(),lr=1e-4,betas=(0.9,0.99))
        self.clip_grad_norm = 10
        
        # training process
        self.net.train()
        for e in range(epochs):
            for iters,sample in enumerate(self.train_loader):
                self.train_step(sample,iters=iters,epoch=e)
            # there is supposed to be a validation func.
            # the model will be saved at every epoch.
            self.save_model(model_name = "net.pkl")
            
    def save_model(self,model_name=None):
        if self.cuda:
            self.net = self.net.cpu()
        if model_name is None:
            model_name = "net.pkl"    
        torch.save(self.net,self.file_path+"/"+model_name)
        if self.cuda:
            self.net = self.net.to(self.device)
        print("Model saved!")
        
    def restore_model(self,model_name=None):
        if model_name is None:
            model_name = "net.pkl"
        self.net = torch.load(self.file_path + "/" + model_name)
        if self.cuda:
            self.net = self.net.to(self.device)
        print("Model restored!")
    
    def __addCoordinates(self,image):
        """
        image:(channel,height,width), torch.float32, sometimes the channel can be n_filters
        """
        image_height,image_width = image.size()[1:]
        
        x_coords = 2.0 * torch.arange(image_height).unsqueeze(1).expand(image_height, image_width) / 255.0 - 1.0
        y_coords = 2.0 * torch.arange(image_width).unsqueeze(0).expand(image_height,image_width) / 255.0 - 1.0
        coords = torch.stack((x_coords, y_coords), dim=0)
        
        image = torch.cat((coords,image),dim=0)
        return image
        
    def instance_predict(self,image,mask=None):
        """
        image is a np.array, [-1,1]. height,width,channel
        """
        image = torch.tensor(image) #(height,width,channel)
        image = image.unsqueeze(0) #(1,height,width,channel)
        
        if self.cuda:
            image = image.to(self.device)
            self.net = self.net.to(self.device)
        with torch.no_grad():
            pred_sem_seg, pred_ins_seg, pred_n_objects_normalized = self.net(image)
        
        # 1.n_objects
        pred_n_objects_normalized = pred_n_objects_normalized.squeeze(0) # (bs,1) -> (1)
        pred_n_objects = pred_n_objects_normalized * self.max_n_objects
        if self.cuda:
            pred_n_objects = pred_n_objects.cpu()
        pred_n_objects = int(np.around(pred_n_objects[0].detach().numpy()))

        # 2.pred_sem_seg
        pred_sem_seg = pred_sem_seg.squeeze(0) # (1,n_classes,height,width) -> (n_classes,height,width) torch.tensor
        if self.cuda:
            pred_sem_seg = pred_sem_seg.cpu()
        pred_sem_seg = pred_sem_seg.detach().numpy()
        pred_sem_seg = pred_sem_seg.argmax(0).astype(np.uint8)
        
        # here the mask is supposed to be defined.
        if mask is None:
            mask = pred_sem_seg
        
        # 3.pred_ins_seg
        pred_ins_seg = pred_ins_seg.squeeze(0) # (1,n_filters,height,width) -> (n_filters,height,width)
        embeddings = self.__addCoordinates(image=pred_ins_seg) # (n_filters,height,width) -> (n_filters+2,height,width)
        if self.cuda:
            embeddings = embeddings.cpu()
        embeddings = embeddings.detach().numpy()
        embeddings = embeddings.transpose(1,2,0)
        embeddings = np.stack([embeddings[:,:,i][mask!=0]
        for i in range(embeddings.shape[2])],axis=1) # pred_sem_seg is used here to provde a mask.
        
        # then cluster
        cluster = SpectralClustering(n_clusters =pred_n_objects,
                                     eigen_solver = None, random_state = None,
                                     n_init = 10, gamma=1.0, affinity = "rbf",
                                     n_neighbors=10, eigen_tol=0.0,
                                     assign_labels= "discretize",degree=3,
                                     coef0=1,kernel_params=None).fit(embeddings)
        labels = cluster.labels_
        # offer the detail about height,width
        height,width = pred_sem_seg.shape[0:]
        instance_mask = np.zeros((height,width),dtype=np.uint8)
        
        fg_coords = np.where(mask!=0)
        for idx in range(len(fg_coords[0])):
            y_coord = fg_coords[0][idx]
            x_coord = fg_coords[1][idx]
            _label = labels[idx] + 1
            instance_mask[y_coord,x_coord] = _label
            
        return pred_sem_seg, instance_mask,pred_n_objects
        
        
        
        