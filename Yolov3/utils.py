# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:53:00 2020

@author: huijian
"""
import torch

# this fcn is used for  build_target_matrix
def bbox_wh_iou(wh1,wh2):
    """
    wh torch.tensor
    wh1: (2)  
    wh2: (n,2)
    
    intersection/union: (n)
    """
    wh2= wh2.t()
    w1,h1 = wh1[0],wh1[1]
    w2,h2 = wh2[0],wh2[1]
    intersection = torch.min(w1,w2) * torch.min(h1,h2)
    union_area = (w1*h1+1e-16) + w2*h2 -intersection
    return intersection / union_area

def bbox_iou(box1,box2, x1y1x2y2 = True):
    """
    bothe box1,box2 should be with a shape of (n,2)
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # xywh
        b1_x1,b1_x2 = box1[:,0] - box1[:,2]/2, box1[:,0] + box1[:,2]/2
        b1_y1,b1_y2 = box1[:,1] - box1[:,3]/2, box1[:,1] + box1[:,3]/2
        b2_x1,b2_x2 = box2[:,0] - box2[:,2]/2, box2[:,0] + box2[:,2]/2 
        b2_y1,b2_y2 = box2[:,1] - box2[:,3]/2, box2[:,1] + box2[:,3]/2
    else:
        # box1/box2 (x1,y1,x2,y2)
        b1_x1,b1_y1,b1_x2,b1_y2 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
        b2_x1,b2_y1,b2_x2,b2_y2 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]
    
    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1,b2_x1)
    inter_rect_y1 = torch.max(b1_y1,b2_y1)
    inter_rect_x2 = torch.min(b1_x2,b2_x2)
    inter_rect_y2 = torch.min(b1_y2,b2_y2)
    
    # Intersection area
    intersection = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,min=0) * torch.clamp(
        inter_rect_y2 -inter_rect_y1 + 1,min=0)
    
    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    #print(intersection)
    #print(b1_area+b2_area-intersection)
    
    iou = intersection / (b1_area + b2_area - intersection + 1e-16)
    
    return iou

def xywh2xyxy(x):
    # from (center_x,center_y,width,height) to (x1,y1,x2,y2)
    # x: [batch_num,sum,4]  4:coordinate
    # x: tensor
    y = x.new(x.shape)
    y[...,0] = x[...,0] - x[...,2]/2
    y[...,1] = x[...,1] - x[...,3]/2
    y[...,2] = x[...,0] + x[...,2]/2
    y[...,3] = x[...,1] + x[...,3]/3
    
    # y (x1,y1,x2,y2):top left (x1,y1)  bottom right (x2,y2)
    return y

def non_max_suppression(prediction,conf_thres=0.5,nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    and performs non-maximum suppression to further filter detections.
    Returns detections with shape.
        (x1,y1,x2,y2,conf_score,cls_score,cls_pred)
    outputs: list and every element in the list has a shape of (detection_num,7)
    """
    # prediction [batch_num,detecion_num,5+cls_num]
    # from xywh2xyxy
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    outputs = [None for _ in range(len(prediction))] # len(output) = batch_num
    
    #print(prediction.shape)
    
    for img_idx,pred in enumerate(prediction):
        #print(pred.shape)
        #a = input()
        #print("img_idx:{}".format(img_idx))
        # pred [sum,5+cls_num]
        # filter out the detections with confidence score below threshold
        # detection_num changes
        pred = pred[pred[:,4]>=conf_thres]
        # if none are remaining, process the next image
        if not pred.size(0):
            continue
        # score = conf_score * cls_score
        # score [detection_num]
        score =pred[:,4]*(pred[:,5:].max(1)[0])
        # sort by score
        # torch.argsort(input,dim=-1,descending=False)
        pred = pred[(-score).argsort()]
        # (values,indices) = torch.max(input,dim,keepdim=False,out=None)
        # cls_score,cls_pred : [detection_num,1]
        cls_score,cls_pred = pred[:,5:].max(1,keepdim=True)
        # detections [detection_num,7]  7: box(4) conf_score(1) cls_score(1) cls_pred(1)
        detections = torch.cat((pred[:,:5],cls_score.float(),cls_pred.float()),1)
        # perform nms
        keep_boxes = []
        # print("nmsing ...")
        # print(nms_thres)
        while detections.size(0):
            # tmp = input()
            # print(detections.size(0))
            #print(detections[0,:4])
            """
            we do not exclude detection detection[0], 
            because we use a kind of soft-nms, which use a fuse strategy.
            """
            #print(bbox_iou(detections[0,:4].unsqueeze(0),detections[:,:4]))
            large_overlap = bbox_iou(detections[0,:4].unsqueeze(0),detections[:,:4]) > nms_thres
            #print(large_overlap)
            label_match = detections[0,-1] == detections[:,-1]
            # indices of boxes with lower confidence scores, large IoUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5] # weights : [detection_num, 1]
            # merge overlapping bboxes by order of confidence
            detections[0,:4] = (weights * detections[invalid,:4]).sum(0) /weights.sum()
            keep_boxes += [detections[0]]
            # delete the invalid detections
            detections = detections[~invalid]
            
        if keep_boxes:
            outputs[img_idx] = torch.stack(keep_boxes)
    return outputs

if __name__=="__main__":
    wh2 = torch.rand(5,2)
    wh1 = torch.rand(2)
    output1= bbox_wh_iou(wh1, wh2)
    
    prediction = torch.rand((4,256,10)) # batch_num=4,cls_num = 5
    output2 = non_max_suppression(prediction) # list
    print(output2[1].shape)