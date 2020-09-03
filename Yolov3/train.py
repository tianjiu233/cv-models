# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:05:09 2020

@author: huijian
"""


import torch
from torch.utils.data import DataLoader

import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from utils import non_max_suppression,xywh2xyxy
from loss_utils import compute_batch_info,ap_per_cls

class Trainer(object):
    def __init__(self,model,
                 model_path,
                 cuda=False,
                 device=torch.device("cpu")):
        
        self.model = model
        self.model_path = model_path
        self.cuda = cuda
        self.device = device
        
    def train_model(self,train_data,val_data,
                    batch_size=4,
                    grad_accumulation=2,
                    eval_interval = 2,
                    epoch_num=int(1e3)):
        
        self.train_data = train_data
        self.val_data = val_data
        
        self.grad_accumulation = grad_accumulation
        self.optim = torch.optim.Adam(self.model.parameters())
        
        self.eval_interval = eval_interval
        
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=batch_size,shuffle=True,
                                       collate_fn=self.train_data.collate_fn)
        
        self.val_loader = DataLoader(dataset=self.val_data,
                                     batch_size=batch_size,shuffle=False,
                                     collate_fn=self.val_data.collate_fn)
        
        if self.cuda:
            self.model = self.model.to(self.device)
        
    
        self.model.train()
        for e in range(epoch_num):
                                    
            for batch_idx,(imgs,labels) in enumerate(tqdm.tqdm(self.train_loader)):
                # check the input_data format
                imgs = imgs.to(torch.float32)
                labels = labels.to(torch.float32)
                if self.cuda:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                
                outputs,loss,metrics = self.model(imgs,labels)
                # accumulate grd
                loss.backward()
                
                # accumulate gradient before each step
                # and then update the parameters
                batches_done = len(self.train_loader)*e+batch_idx
                if batches_done % self.grad_accumulation == 0:
                    self.optim.step()
                    self.optim.zero_grad()  
                    
            # log the info
            print("Info During training...")
            print("epoch-{}".format(str(e+1)))
            print("---small---")
            tmp = metrics[0]
            print("stride:{}".format(tmp["stride"]))
            print("total_loss:{:.3f}".format(tmp["loss"]))
            print("loss_x:{:.3f},loss_y:{:.3f},loss_w:{:.3f},loss_h:{:.3f}".format(tmp["x"],tmp["y"],tmp["h"],tmp["w"]))
            print("loss_conf:{:.3f},loss_cls:{:.3f}".format(tmp["conf"],tmp["cls"]))
            print("cls_accu:{:.3f}".format(tmp["cls_accu"]))
            print("recall50:{:.3f},recall75:{:.3f},precision:{:.3f}".format(tmp["recall50"],tmp["recall75"],tmp["precision"]))
            print("conf_obj:{:.3f},conf_noobj:{:.3f}".format(tmp["conf_obj"],tmp["conf_noobj"]))
            
            print("---medium---")
            tmp = metrics[1]
            print("stride:{}".format(tmp["stride"]))
            print("total_loss:{:.3f}".format(tmp["loss"]))
            print("loss_x:{:.3f},loss_y:{:.3f},loss_w:{:.3f},loss_h:{:.3f}".format(tmp["x"],tmp["y"],tmp["h"],tmp["w"]))
            print("loss_conf:{:.3f},loss_cls:{:.3f}".format(tmp["conf"],tmp["cls"]))
            print("cls_accu:{:.3f}".format(tmp["cls_accu"]))
            print("recall50:{:.3f},recall75:{:.3f},precision:{:.3f}".format(tmp["recall50"],tmp["recall75"],tmp["precision"]))
            print("conf_obj:{:.3f},conf_noobj:{:.3f}".format(tmp["conf_obj"],tmp["conf_noobj"]))
            
            print("---large---")
            tmp = metrics[2]
            print("stride:{}".format(tmp["stride"]))
            print("total_loss:{:.3f}".format(tmp["loss"]))
            print("loss_x:{:.3f},loss_y:{:.3f},loss_w:{:.3f},loss_h:{:.3f}".format(tmp["x"],tmp["y"],tmp["h"],tmp["w"]))
            print("loss_conf:{:.3f},loss_cls:{:.3f}".format(tmp["conf"],tmp["cls"]))
            print("cls_accu:{:.3f}".format(tmp["cls_accu"]))
            print("recall50:{:.3f},recall75:{:.3f},precision:{:.3f}".format(tmp["recall50"],tmp["recall75"],tmp["precision"]))
            print("conf_obj:{:.3f},conf_noobj:{:.3f}".format(tmp["conf_obj"],tmp["conf_noobj"]))
            
            # validate the model
            if (e+1) % self.eval_interval==0:
                precision,recall,ap,f1,ap_cls = self.val_model(self.val_loader)
                # log the validate info
                print("validate-epoch-{}".format(str(e+1)))
                print("The map is:{:.3f}".format(ap.mean()))
                print("The ap is")
                for idx,cls_name in enumerate(ap_cls):
                    print("class-{}:ap-{:.3f}".format(str(ap_cls[idx]),ap[idx]))
                model_name = "yolo_" + str(e+1) + ".pkl"
                self._save_model(model_name = model_name)
                
    def restore_model(self,model_name="yolo.pkl"):
        self.model = torch.load(self.model_path+"/"+model_name,map_location=torch.device("cpu"))
        if self.cuda:
            self.model = self.model.to(self.device)
        print("model restored!")
        
    def _save_model(self,model_name="yolo.pkl"):
        if self.cuda:
            self.model = self.model.to(torch.device("cpu"))
        torch.save(self.model,self.model_path+"/"+model_name)
        if self.cuda:
            self.model = self.model.to(self.device)
        print("model saved!")
    
    def visualize_val_data(self,val_data):
        self.model.eval()
        dataloader = DataLoader(dataset=val_data,batch_size=1,shuffle=False,
                                collate_fn=val_data.collate_fn)
        for batch_idx,(img,_) in enumerate(tqdm.tqdm(dataloader)):
            
            if self.cuda:
                img = img.to(torch.float32)
                img = img.to(self.device)
            # outputs are now of real position not related position
            with torch.no_grad():
                outputs,_,__ = self.model(img)
            # for batch_num is 1, len(outputs) == 1 and outputs is a tensor
            # outputs are fed into nms.
            # and outputs are from xywhxyxy
            outputs = non_max_suppression(outputs,conf_thres=0.5,nms_thres=0.4)
            # outputs:[detection_num,7] (x1,y1,x2,y2,conf_score,cls_score,cls_pred)
            outputs = outputs[0] # outputs is a list here, so we get the first one(also the only one)
            
            # visualize
            if self.cuda:
                img = img.cpu()
                
            img = img.squeeze(0)
            img = img.numpy().transpose(1,2,0)
            img = (img*255).astype(np.int) # np.array 0-255
            
            # visualize the img
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(idx) for idx in np.linspace(0,1,20)]
            # plt.figure()
            fig,ax = plt.subplots(1)
            ax.imshow(img)
            
            if outputs is not None:
                
                if self.cuda:
                    outputs = outputs.cpu()
                
                outputs = outputs.numpy()
                bbox_colors = random.sample(colors,len(val_data.cls_dict))
                
                for idx in range(len(outputs)):
                    xmin,ymin,xmax,ymax,_,__,cls_name = outputs[idx][:]
                    
                    box_w = xmax-xmin
                    box_h = ymax-ymin
                    
                    color = bbox_colors[int(cls_name)]
                    bbox = patches.Rectangle((xmin,ymin), width=box_w, height=box_h,
                                         linewidth=2,edgecolor=color,facecolor="none")
                    ax.add_patch(bbox)
                    plt.text(
                    xmin,ymin,
                    s=val_data.cls_dict[int(cls_name)],
                    color = "White",
                    verticalalignment="top",
                    bbox = {"color":color,"pad":0},
                    )
            
            # save the img every epoch
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = "./result/" + "img_idx_" + str(batch_idx+1) + ".png"
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
        
        return 
                    
            
    def val_model(self,val_dataloader,
                  iou_thres=0.5,
                  conf_thres=0.5,
                  nms_thres=0.5,):
        print("validating...")
        self.model.eval()
        
        cls_list = []
        metrics_list = [] # list of tuples (tp,confs,pred)
        for batch_idx,(imgs,labels) in enumerate(tqdm.tqdm(val_dataloader,desc="Detecting objects")):
            
            # check the input_data format
            imgs = imgs.to(torch.float32)
            labels = labels.to(torch.float32)
            if self.cuda:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
            
            # extract cls_name
            # labels: [detection_num,6]- 6:(1) img_id (corresponding to batch_idx) (2) cls_name (4) boxes
            # every item in outputs [detection_num,7] :(x1,y1,x2,y2,conf_score,cls_score,cls_pred)
            cls_list += labels[:,1].tolist()
            
            # rescale labels
            img_h = imgs.size(2)
            img_w = imgs.size(3)
            labels[:,2:] = xywh2xyxy(labels[:,2:])
            labels[:,2] *= img_w
            labels[:,4] *= img_w
            labels[:,3] *= img_h
            labels[:,5] *= img_h
                
                # from xywh->xyxy
            with torch.no_grad():
                outputs,_,__ = self.model(imgs)
                outputs = non_max_suppression(outputs,conf_thres,nms_thres)
            
                # before the ouputs are fed into compute_batch_info fcn
                # the outputs are supposed to be rescaled.
                # metrics_list: tp, pred_conf, pred_cls
                """
                to check the compute_batch_info fcn
                we try to build a new outputs tensor from label to make.
                And in theory, you will get a prefect result.
                """
                """
                FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
                fake_outputs = [None for idx in range(len(outputs))]
                for i in range(len(outputs)):
                    label = labels[labels[:,0]==i]
                    if len(label) > 0:
                        fake_output = FloatTensor(np.zeros((len(label),7)))
                        fake_output[:,:4] = label[:,2:6]
                        fake_output[:,4] = 0.8
                        fake_output[:,5] = 0.8
                        fake_output[:,6] = label[:,1]
                    fake_outputs[i] = fake_output
                outputs = fake_outputs
                """
                metrics_list += compute_batch_info(outputs,labels,iou_thres)
            # for debug
            if batch_idx == 107:
                break
            
        # concatenate sample statistics
        tp,pred_conf,pred_cls = [np.concatenate(x,0) for x in list(zip(*metrics_list))]
        # print(tp.shape)
        # print(pred_conf.shape)
        # print(pred_cls.shape)
        # print(len(np.unique(pred_cls)))
        # a = input()
        precision,recall,ap,f1,ap_cls = ap_per_cls(tp,pred_conf,pred_cls,cls_list)
        # print(precision)
        # print(recall)
        # print(ap.shape)
        # print(f1)
        # print(ap_cls.shape)
        # a =input()
        self.model.train()
    
        return precision,recall,ap,f1,ap_cls
    