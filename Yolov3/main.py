# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:09:37 2020

@author: huijian
"""


import numpy as np
import torch
from torch.utils.data import DataLoader

from yolo import YoloV3

from train import Trainer
from hat_data.hardhat import HardHat


if __name__=="__main__":
    
    # (1) set the data
    annotation_path = "./GDUT-HWD/Annotations/"
    img_path = "./GDUT-HWD/JPEGImages/"
    label_path = "./GDUT-HWD/labels/"
    trainval_path = "./GDUT-HWD/ImageSets/Main/trainval.txt"
    test_path = "./GDUT-HWD/ImageSets/Main/test.txt"
    
    img_shape = 512
    
    train_data = HardHat(ann_path = annotation_path,
                         img_path = img_path,
                         file_name = trainval_path,
                         img_shape = img_shape,
                         augment = True,
                         multi_scale = True)
    
    val_data = HardHat(ann_path = annotation_path,
                       img_path = img_path,
                       file_name = test_path,
                       img_shape = img_shape,
                       multi_scale = False)
    
    # (2) model
    # model hyparameters
    in_chs = 3
    cls_num = 5
    
    yolo = YoloV3(in_chs,cls_num,
                  small_anchors = [[10,13],[16,30],[33,23]],
                  medium_anchors = [[30,61],[62,45],[59,119]],
                  large_anchors = [[116,90],[156,198],[373,326]])
    
    # (3) build a trainer
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model_path = "./checkpoint"
    trainer = Trainer(model=yolo,cuda=cuda,device=device,model_path = model_path)
    if True:
        trainer.restore_model(model_name = "yolo.pkl")
    
    # (4) train the model
    # train hyparameters
    batch_size = 6
    grad_accumulation = 1
    eval_interval = 4
    epoch_num = int(1e3)
    if False:
        trainer.train_model(train_data=train_data,
                            val_data=val_data,
                            batch_size=batch_size,
                            grad_accumulation=grad_accumulation,
                            eval_interval = eval_interval,
                            epoch_num=epoch_num)
        
    
    # (5) eval the model
    # val_model hyparamters
    if False:
        iou_thres = 0.5
        conf_thres = 0.5
        nms_thres = 0.5
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=2,shuffle=False,
                                    collate_fn=val_data.collate_fn)
        precision,recall,ap,f1,ap_cls=trainer.val_model(val_dataloader,iou_thres,conf_thres,nms_thres)
        print("The map is:{:.3f}".format(ap.mean()))
        print("The ap is")
        for idx,cls_name in enumerate(ap_cls):
            print("class-{}:ap-{:.3f}".format(str(ap_cls[idx]),ap[idx]))
        
    # visualizing
    if True:
        trainer.visualize_val_data(val_data)
        
    
    
    

    
    
    