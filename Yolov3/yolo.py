# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:58:45 2020

@author: huijian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_utils import label2target

# these libs are used for test.
from hat_data.hardhat import HardHat
from torch.utils.data import DataLoader

class Tensor2Detection(nn.Module):
    def __init__(self,cls_num,stride,
                 anchors,
                 ignore_thres = 0.5,
                 obj_w = 1,
                 noobj_w = 100):
        super(Tensor2Detection,self).__init__()
        """
        img_shape:(height,width)
        stride:int
        anchors:[]
        """
        self.cls_num = cls_num
        
        self.anchors = anchors  # [[anchor_w1,anchor_h1],...,[anchor_wn,anchor_hn]] In yolov3, n=3.
        self.stride = stride
        
        self.ignore_thres =ignore_thres
        self.obj_w = obj_w
        self.noobj_w = noobj_w
        
        # this attribute will change when we get input_tensor
        self.cuda =False
        
        # set loss fcns
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def compute_grid_offset(self,grid_x,grid_y):
        """
        The aim of this fcn is to compute the grid_offset, which will be used in forward fcn 
        (no matter label is None or not)
        and also the anchor  (scaled anchor, scaled anchor w/h) are computed.
        """
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor   
        
        self.grid_width = grid_x
        self.grid_height = grid_y
        
        # tensor
        self.grid_x = torch.arange(grid_x).repeat(grid_y,1).view([1,1,grid_y,grid_x]).type(FloatTensor) 
        # tensor
        self.grid_y = torch.arange(grid_y).repeat(grid_x,1).t().view([1,1,grid_y,grid_x]).type(FloatTensor)
        # tensor
        # self.scaled_anchors [n,2]
        self.scaled_anchors =FloatTensor([(anchor_w/self.stride,anchor_h/self.stride) for anchor_w,anchor_h in self.anchors])
        # self.scaled_anchor_w/h [1,n,1,1]
        self.scaled_anchor_w = self.scaled_anchors[:,0:1].view((1,len(self.anchors),1,1))
        self.scaled_anchor_h = self.scaled_anchors[:,1:2].view((1,len(self.anchors),1,1))
        
    def forward(self,input_tensor,label=None):
        # input_tensor [batch_num,out_chs,gird_y,grid_x]
        
        batch_num = input_tensor.size(0)
        grid_y = input_tensor.size(2) # height(row)
        grid_x = input_tensor.size(3) # width(column)
        
        # For we may set multi-scale True, we must compute_grid_offset at every batch
        self.cuda = input_tensor.is_cuda
        self.compute_grid_offset(grid_x=grid_x,grid_y=grid_y)
        
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        #LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        #ByteTensor = torch.cuda.ByteTensor if self.cuda else torch.ByteTensor
        
        
        # (batch_num,out_chs,height,width) -> (bathc_num,anchor_num,cls_num+5,grid_height,grid_width)
        # then
        # (batch_num,anchor_num,cls_num+5,grid_height,grid_width) -> (batch_num,anchor_num,gird_height,gird_width,num_cls+5)
        
        # out_chs = (cls_num + 5)*anchor_num
        # height = grid_height // width = grid_width
        # prediction [batch_num,anchor_num,grid_height,gird_width,num_cls+5]
        # the permute fcn changes the place/order of self.cls_num+5
        prediction = (
            input_tensor.view(batch_num,len(self.anchors),self.cls_num+5,self.grid_height,self.grid_width).
            permute(0,1,3,4,2).
            contiguous())
        
        # the order of cls_num+5 is (4) boxes, (1) pred_conf, (cls_num) cls_conf
        # activation
        pred_x = torch.sigmoid(prediction[...,0]) # center x (actually tx)
        pred_y = torch.sigmoid(prediction[...,1]) # center y
        pred_w = prediction[...,2] # width
        pred_h = prediction[...,3] # height
        pred_conf = torch.sigmoid(prediction[...,4]) # confidence score
        cls_conf = torch.sigmoid(prediction[...,5:]) # cls score
        
        # add offset and scale with anchors
        # pred_boxes [batch_num,anchor_num,grid_height,grid_width,4]
        pred_boxes = FloatTensor(prediction[...,:4].shape)
        pred_boxes[...,0] = pred_x + self.grid_x
        pred_boxes[...,1] = pred_y + self.grid_y
        pred_boxes[...,2] = torch.exp(pred_w) * self.scaled_anchor_w
        pred_boxes[...,3] = torch.exp(pred_h) * self.scaled_anchor_h
        
        # output_tensor [batch_num,gird_y*grid_x*anchor_num,5+cls_num]
        # xywh
        output_tensor = torch.cat(
            (
            pred_boxes.view(batch_num,-1,4) * self.stride, 
            pred_conf.view(batch_num,-1,1),
            cls_conf.view(batch_num,-1,self.cls_num)
            ),-1)
        
        # label is None, when it is the inference.
        if label is None:
            return output_tensor,0,None
        
        # label is not None, then compute the loss and return output and loss
        # the anchors that required here should be the scaled ones, which is corresponding the feature map
        iou_scores,cls_mask,obj_mask,noobj_mask,tx,ty,tw,th,tcls,tconf = label2target(pred_boxes=pred_boxes, 
                                                                                      cls_conf=cls_conf,
                                                                                      label=label, 
                                                                                      anchors=self.scaled_anchors, 
                                                                                      ignore_thres=self.ignore_thres)
        
        
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()
        # loss
        loss_x = self.mse_loss(pred_x[obj_mask],tx[obj_mask])
        loss_y = self.mse_loss(pred_y[obj_mask],ty[obj_mask])
        loss_w = self.mse_loss(pred_w[obj_mask],tw[obj_mask])
        loss_h = self.mse_loss(pred_h[obj_mask],th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask],tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask],tconf[noobj_mask])
        loss_conf = self.obj_w * loss_conf_obj + self.noobj_w * loss_conf_noobj
        loss_clf = self.bce_loss(cls_conf[obj_mask],tcls[obj_mask])
        loss = loss_x+loss_y+loss_h+loss_w + loss_conf+loss_clf
        
        # metrics
        cls_accu = 100 * cls_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf>0.5).float()
        iou50 =(iou_scores>0.5).float()
        iou75 =(iou_scores>0.75).float()
        
        # tconf is the same as obj_mask
        detected_mask = conf50*cls_mask*tconf
        precision = torch.sum(iou50*detected_mask) / (conf50.sum()+1e-16)
        recall50 = torch.sum(iou50*detected_mask) / (obj_mask.sum()+1e-16)
        recall75 = torch.sum(iou75*detected_mask) / (obj_mask.sum()+1e-16)

        metrics = {
            "loss":float(loss.cpu().detach().numpy()),
            "x":float(loss_x.cpu().detach().numpy()),
            "y":float(loss_y.cpu().detach().numpy()),
            "w":float(loss_w.cpu().detach().numpy()),
            "h":float(loss_h.cpu().detach().numpy()),
            "conf":float(loss_conf.cpu().detach().numpy()),
            "cls":float(loss_clf.cpu().detach().numpy()),
            "cls_accu":float(cls_accu.cpu().detach().numpy()),
            "recall50":float(recall50.cpu().detach().numpy()),
            "recall75":float(recall75.cpu().detach().numpy()),
            "precision":float(precision.cpu().detach().numpy()),
            "conf_obj":float(conf_obj.cpu().detach().numpy()),
            "conf_noobj":float(conf_noobj.cpu().detach().numpy()),
            "stride":int(self.stride)
            }

        return output_tensor,loss,metrics


class ConvOp(nn.Module):
    def __init__(self,in_chs,out_chs,
                 kernel_size=3,stride=1,
                 pad=1,dilation=1,bias=False):
        super(ConvOp,self).__init__()
        self.op = nn.Sequential(nn.Conv2d(in_channels=in_chs,out_channels=out_chs,
                                          kernel_size=kernel_size,stride=stride,
                                          padding=pad,dilation=dilation,
                                          bias=bias),
                                nn.BatchNorm2d(out_chs,momentum=0.9,eps=1e-5),
                                nn.LeakyReLU(0.1))
    def forward(self,input_tensor):
        output_tensor = self.op(input_tensor)
        return output_tensor

class ResBlock(nn.Module):
    def __init__(self,in_chs,mid_chs):
        super(ResBlock,self).__init__()
        self.op1 = ConvOp(in_chs=in_chs, out_chs=mid_chs,kernel_size=1,pad=0)
        self.op2 = ConvOp(in_chs=mid_chs, out_chs=in_chs,kernel_size=3,pad=1)
    
    def forward(self,input_tensor):
        x = self.op1(input_tensor)
        x = self.op2(x)
        output_tensor = x+input_tensor
        return output_tensor
        
class DarkNet(nn.Module):
    def __init__(self,in_chs=3):
        super(DarkNet,self).__init__()
        self.pre_conv = nn.Sequential(ConvOp(in_chs=in_chs,out_chs=32),
                                      ConvOp(in_chs=32,out_chs=64,stride=2))
        
        # resblock: in_chs=out_chs
        # 1x
        self.resblock1 = ResBlock(in_chs=64, mid_chs=32)
        
        self.down_conv1 = ConvOp(in_chs=64, out_chs=128,stride=2)
        
        # 2x
        self.resblock2_1 = ResBlock(in_chs=128, mid_chs=64)
        self.resblock2_2 = ResBlock(in_chs=128, mid_chs=64)
        
        self.down_conv2 = ConvOp(in_chs=128, out_chs=256,stride=2)
        
        # 8x
        self.resblock3_1 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_2 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_3 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_4 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_5 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_6 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_7 = ResBlock(in_chs=256, mid_chs=128)
        self.resblock3_8 = ResBlock(in_chs=256, mid_chs=128)
        
        self.down_conv3 = ConvOp(in_chs=256,out_chs=512,stride=2)
        
        # 8x
        self.resblock4_1 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_2 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_3 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_4 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_5 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_6 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_7 = ResBlock(in_chs=512, mid_chs=256)
        self.resblock4_8 = ResBlock(in_chs=512, mid_chs=256)
        
        self.down_conv4 = ConvOp(in_chs=512, out_chs=1024,stride=2)
        
        # 4x
        self.resblock5_1 = ResBlock(in_chs=1024, mid_chs=512)
        self.resblock5_2 = ResBlock(in_chs=1024, mid_chs=512)
        self.resblock5_3 = ResBlock(in_chs=1024, mid_chs=512)
        self.resblock5_4 = ResBlock(in_chs=1024, mid_chs=512)
        
    def forward(self,input_tensor):
        
        # save the features maps
        tensors = []
        
        x = self.pre_conv(input_tensor)  # 256->128
        
        # res-down-1
        x = self.resblock1(x)
        x = self.down_conv1(x) # 128->64
        
        # res-down-2
        # res
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        # down
        x = self.down_conv2(x) # 64->32
        
        # res-down-3
        # res
        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        x = self.resblock3_3(x)
        x = self.resblock3_4(x)
        x = self.resblock3_5(x)
        x = self.resblock3_6(x)
        x = self.resblock3_7(x)
        x = self.resblock3_8(x)
        # save to feat_maps
        tensors.append(x)  # 32x32  1/8
        # down
        x = self.down_conv3(x) # 32->16
        
        # res-down-4
        # res
        x = self.resblock4_1(x)
        x = self.resblock4_2(x)
        x = self.resblock4_3(x)
        x = self.resblock4_4(x)
        x = self.resblock4_5(x)
        x = self.resblock4_6(x)
        x = self.resblock4_7(x)
        x = self.resblock4_8(x)
        # save to feat_maps
        tensors.append(x) # 16x16  1/16
        # down 
        x = self.down_conv4(x) # 16->8
        
        # res-5
        # res
        x = self.resblock5_1(x)
        x = self.resblock5_2(x)
        x = self.resblock5_3(x)
        x = self.resblock5_4(x)
        # save to feat_maps
        tensors.append(x) # 8x8 1/32
        
        return tensors # [1/8,1/16,1/32]

class ConvOpBlock(nn.Module):
    """
    This module is used in YoloV3.
    """
    def __init__(self,in_chs,out_chs,mid_chs):
        super(ConvOpBlock,self).__init__()
        self.op = nn.Sequential(ConvOp(in_chs=in_chs,out_chs=out_chs,kernel_size=1,pad=0),
                                ConvOp(in_chs=out_chs,out_chs=mid_chs),
                                ConvOp(in_chs=mid_chs,out_chs=out_chs,kernel_size=1,pad=0),
                                ConvOp(in_chs=out_chs,out_chs=mid_chs),
                                ConvOp(in_chs=mid_chs,out_chs=out_chs,kernel_size=1,pad=0))
    def forward(self,input_tensor):
        output_tensor = self.op(input_tensor)
        return output_tensor

class UpsampleLayer(nn.Module):
    def __init__(self,scale_factor,mode="nearest"):
        super(UpsampleLayer,self).__init__()
        self.scale_factor= scale_factor
        self.mode = mode
    def forward(self,input_tensor):
        output_tensor = F.interpolate(input_tensor,
                                      scale_factor=self.scale_factor,mode=self.mode)
        return output_tensor

class YoloV3(nn.Module):
    def __init__(self,in_chs,cls_num,
                 small_anchors = [[10,13],[16,30],[33,23]],
                 medium_anchors = [[30,61],[62,45],[59,119]],
                 large_anchors = [[116,90],[156,198],[373,326]]):
        
        super(YoloV3,self).__init__()
        
        # compute hyparametes:        
        self.small_anchors = small_anchors
        self.medium_anchors = medium_anchors
        self.large_anchors = large_anchors

        # anchor num for per feature map
        anchor_num = len(self.small_anchors)
        # out_chs 5: (4): boxes (1) pred_conf 
        out_chs = (5+cls_num)*anchor_num # 3 anchors for each feature map
        
        # the network structure
        self.backbone = DarkNet(in_chs=in_chs) # tensor:[256,512,1024]
        
        self.convop_block1 = ConvOpBlock(in_chs=1024,out_chs=512, mid_chs=1024) 
        self.convop_block2 = ConvOpBlock(in_chs=512+256,out_chs=256,mid_chs=512) 
        self.convop_block3 = ConvOpBlock(in_chs=256+128, out_chs=128, mid_chs=256)
        
        self.upsample1 = nn.Sequential(ConvOp(in_chs=512,out_chs=256,kernel_size=1,pad=0),
                                       UpsampleLayer(scale_factor=2))
        self.upsample2 = nn.Sequential(ConvOp(in_chs=256,out_chs=128,kernel_size=1,pad=0),
                                       UpsampleLayer(scale_factor=2))
        
        self.conv1 = nn.Sequential(ConvOp(in_chs=512, out_chs=1024),
                                   nn.Conv2d(in_channels=1024, out_channels=out_chs, kernel_size=1,padding=0))
        self.conv2 = nn.Sequential(ConvOp(in_chs=256, out_chs=512),
                                   nn.Conv2d(in_channels=512, out_channels=out_chs, kernel_size=1,padding=0))
        self.conv3 = nn.Sequential(ConvOp(in_chs=128, out_chs=256),
                                   nn.Conv2d(in_channels=256, out_channels=out_chs, kernel_size=1,padding=0))
        
        
        
        
        # the detection layers
        # stride = 8
        self.small_detection_layer = Tensor2Detection(cls_num=cls_num, 
                                                      stride=8, 
                                                      anchors=self.small_anchors,
                                                      ignore_thres = 0.5,
                                                      obj_w = 1,
                                                      noobj_w = 100)
        # stride = 16
        self.medium_detection_layer = Tensor2Detection(cls_num=cls_num, 
                                                       stride=16, 
                                                       anchors=self.medium_anchors,
                                                       ignore_thres = 0.5,
                                                       obj_w = 1,
                                                       noobj_w = 100)
        # stride = 32
        self.large_detection_layer = Tensor2Detection(cls_num=cls_num, 
                                                      stride=32, 
                                                      anchors=self.large_anchors,
                                                      ignore_thres = 0.5,
                                                      obj_w = 1,
                                                      noobj_w = 100)
        
    def forward(self,input_tensor,label=None):
        
        #tensors = []
        
        small,medium,large = self.backbone(input_tensor)
        
        # large objects
        x = self.convop_block1(large) # 1024->512
        large_outputs = self.conv1(x) # 512->out_chs
        #tensors.append(large_detections)
        
        # middle objects
        x = self.upsample1(x) # 512->256
        x = torch.cat((x,medium),1) # 256->256+512
        x = self.convop_block2(x) # 256+512->256
        medium_outputs = self.conv2(x) # 256->out_chs
        #tensors.append(medium_detections)  
        
        # small objects
        x = self.upsample2(x) # 256->128
        x = torch.cat((x,small),1) # 128->128+256
        x = self.convop_block3(x) # 128+256->128
        small_outputs = self.conv3(x) # 128->out_chs
        
        # the below is the special part of detection task.
        
        # then the tensors will be transfer to detections
        small_outputs,small_loss,small_metrics = self.small_detection_layer(small_outputs,label)
        medium_outputs,medium_loss,medium_metrics = self.medium_detection_layer(medium_outputs,label)
        large_outputs,large_loss,large_metrics = self.large_detection_layer(large_outputs,label)
        
        # concat all the detections
        """
        xxx_outputs is with a shape of [batch_num,anchors_num*grid_y*grid_x,5+cls_num]
        when concating, the result should be with a shape of
        [batch_num,detection_num,5+cls_num]
        """
        outputs = torch.cat((small_outputs,medium_outputs,large_outputs),dim=1)
        loss = small_loss + medium_loss + large_loss
        metrics = [small_metrics,medium_metrics,large_metrics]
        return outputs,loss,metrics

if __name__=="__main__":
    print("Testing DarkNet and Yolo...")
    
    sample = torch.rand((1,3,512,512))
    
    cuda = torch.cuda.is_available()
    
    darknet = DarkNet(in_chs=3)
    with torch.no_grad():
        tensors = darknet(sample)
        print(len(tensors))
        for item in tensors:
            print(type(item))
            print(item.shape)
    
    yolo = YoloV3(in_chs=3,cls_num=5)
    with torch.no_grad():
        outputs,loss,metrics = yolo(sample,label=None)
        print(outputs.shape)
        print(loss)
        print(metrics)
        
    annotation_path = "./GDUT-HWD/Annotations/"
    img_path = "./GDUT-HWD/JPEGImages/"
    label_path = "./GDUT-HWD/labels/"
    trainval_path = "./GDUT-HWD/ImageSets/Main/trainval.txt"
    test_path = "./GDUT-HWD/ImageSets/Main/test.txt"
    
    hats = HardHat(ann_path = annotation_path,
                   img_path = img_path,
                   file_name = trainval_path,
                   img_shape = 576,
                   augment = False,
                   multi_scale = False)
    
    hats._vis_sample(index=123)
    
    hats_dataloader = DataLoader(dataset=hats,batch_size=4,shuffle=True,
                                 collate_fn=hats.collate_fn)
    
    # when label is not none
    for idx,(img,label) in enumerate(hats_dataloader):
        print("This is batch_{}".format(idx))
        if idx == 6:
            #print(label.shape)
            #print(label[:,0])
            print(img.shape)
            print(label.shape)
            outputs,loss,metrics = yolo(input_tensor=img,label=label)
            print(outputs.shape)
            print(loss)
            print(metrics)
            break
    
    # when label is none
    with torch.no_grad():
        outputs,loss,metrics = yolo(input_tensor =img)
    