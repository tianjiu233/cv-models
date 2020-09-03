# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:16:29 2020

@author: huijian
"""

import torch
import numpy as np
import tqdm

from utils import bbox_iou,bbox_wh_iou

"""
This fcn is important.
With this fcn, we get the target from label.
It means for every feature_map, there should be a target (matrix)
"""
def label2target(pred_boxes,cls_conf,
                 label,
                 anchors,
                 ignore_thres):
    """
    anchors: the scaled_anchors
    pred_boxes: [batch_num,anchor_num,grid_height,grid_width,4]
    cls_conf: [batch_num,anchor_num,grid_height,grid_width,cls_num]
    label:[detection_num,6] # 6: img_id(1), cls_name(1), boxes(4) (cx,cy,w,h)
    """
    # set cuda
    cuda = pred_boxes.is_cuda
    
    ByteTensor = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # pred_boxes [batch_num,anchor_num,grid_height,grid_width,4]
    batch_num= pred_boxes.size(0)
    anchor_num = pred_boxes.size(1)
    # usually grid_height==grid_width
    grid_height = pred_boxes.size(2)
    grid_width = pred_boxes.size(3)
    # print("gird_width:{}, and grid_height:{}".format(grid_height,grid_width))
    # cls_conf: the predicted cls confidence scores
    # cls_conf [batch_num,anchor_num,grid_height,grid_width,cls_num]
    cls_num = cls_conf.size(-1)
    
    # output_tensor 
    # Most output_tensor is of a shape [batch_num,anchor_num,grid_height,grid_width]
    obj_mask =ByteTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    noobj_mask = ByteTensor(batch_num,anchor_num,grid_height,grid_width).fill_(1)
    cls_mask = FloatTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    iou_scores = FloatTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    """
    tx: target x
    the 't' here means target , and it also can be explained as truth.
    """
    tx =FloatTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    ty =FloatTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    tw =FloatTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    th =FloatTensor(batch_num,anchor_num,grid_height,grid_width).fill_(0)
    # tcls and tconf are the special cases
    tcls = FloatTensor(batch_num,anchor_num,grid_height,grid_width,cls_num).fill_(0)
    tconf = None
    
    # label [detection_num,6]
    # it should be noticed that 
    # for label (6): (1)img_id (1) cls_name (4) boxes xywh[0-1]
    # for output_tensot (5+cls_num) : (4) boxes (1) pred_conf (cls_num) cls_conf
    """
    It's is very important that we use the *1.here, this will help us to keep label unchanged.
    Or you can say that it will seperate the true_boxes and label
    """
    true_boxes = label[:,2:6]*1. # [detection_num,4] (center_x,center_y,width,height)
    # print("true_boxes_max:{}".format(true_boxes.max()))
    # conver to postion relative to box
    true_boxes[:,0] = true_boxes[:,0]*grid_width
    true_boxes[:,2] = true_boxes[:,2]*grid_width
    true_boxes[:,1] = true_boxes[:,1]*grid_height
    true_boxes[:,3] = true_boxes[:,3]*grid_height
    # true_xy,true_wh has a shape of [detection_num,2]
    true_xy = true_boxes[:,:2] # center_x,center_y
    true_wh = true_boxes[:,2:] # width,height
    # get anchors with best iou
    # torch.stack(tensor,dim=0,out=None)
    # ious [anchor_num,detection_num]
    
    ious = torch.stack([bbox_wh_iou(anchor,true_wh) for anchor in anchors])
    # use fcn max() to find the best anchor (iou_value and anchor_idx)
    # both best_ious and best_anchor_id have the shape of [detection_num]
    best_ious, best_anchor_idx = ious.max(0)
    # both img_idx and cls_name have a shape of [detection_num]
    img_idx,cls_name = label[:,:2].long().t()
    
    # all the tensor has a shape of [detection]
    true_x,true_y = true_xy.t()
    true_w,true_h = true_wh.t()
    coor_x,coor_y = true_xy.long().t()
    
    # set masks
    """
    print("--------")
    print(obj_mask.shape)
    print(best_anchor_idx.max())
    print(coor_y.max())
    print(coor_x.max())
    """
    
    # img_idx = img_idx.long()
    # best_anchor_idx = best_anchor_idx.long()
    # coor_y = coor_y.long()
    # coor_x = coor_x.long()
    
    obj_mask[img_idx,best_anchor_idx,coor_y,coor_x] = 1
    noobj_mask[img_idx,best_anchor_idx,coor_y,coor_x] = 0
    
    # set noobj mask to zero where iou exceeds ignore threshold
    # ious.t() (detection_num, anchor_num)
    # Except for the true or the ideal boxes, there are some boxes also meeting the requirements.
    # Thus, we set noobj_mask of these boxes to be 0.
    for idx,anchor_iou in enumerate(ious.t()):
        noobj_mask[img_idx[idx],anchor_iou>ignore_thres,coor_y[idx],coor_x[idx]] = 0
    
    # coordinates
    tx[img_idx,best_anchor_idx,coor_y,coor_x] = true_x - true_x.floor() # from cx to tx
    ty[img_idx,best_anchor_idx,coor_y,coor_x] = true_y - true_y.floor()
    tw[img_idx,best_anchor_idx,coor_y,coor_x] = torch.log(true_w/anchors[best_anchor_idx][:,0]+1e-16)
    th[img_idx,best_anchor_idx,coor_y,coor_x] = torch.log(true_h/anchors[best_anchor_idx][:,1]+1e-16)
    
    # one-hot labeling 
    tcls[img_idx,best_anchor_idx,coor_y,coor_x,cls_name] = 1
    
    # compute label correctness and iou at best anchor idx
    # cls_mask: the element in the matrix is 1, when the predicted label is true
    # argmax(-1) corresponding the final dim
    cls_mask[img_idx,best_anchor_idx,coor_y,coor_x] = (cls_conf[img_idx,best_anchor_idx,coor_y,coor_x].argmax(-1)==cls_name).float()
    iou_scores[img_idx,best_anchor_idx,coor_y,coor_x] = bbox_iou(pred_boxes[img_idx,best_anchor_idx,coor_y,coor_x],
                                                                 true_boxes,x1y1x2y2=False)
    tconf = obj_mask.float()
    
    return iou_scores,cls_mask,obj_mask,noobj_mask,tx,ty,tw,th,tcls,tconf
    

def compute_batch_info(outputs,labels,iou_thres):
    """
    compute true positive, predicted scores and predicted labels per sample batch
    here, the outputs are the concated outputs.
    """
    batch_metrics = []
    for idx in range(len(outputs)):
        # idx is img_idx
        """
        Before outputs flows into compute_batch_info fcn,
        the outputs should be processed by nms, and outputs will be a list.
        len(outputs) == batch_num
        So it may exist None.
        every item in outputs: [detection_num,7]
        (x1,y1,x2,y2,conf_score,cls_score,cls_pred)
        """
        if outputs[idx] is None:
            continue
        
        output = outputs[idx]
        pred_boxes = output[:,:4]
        pred_conf = output[:,4]
        pred_cls = output[:,-1]
        
        # true positive
        tp = np.zeros(pred_boxes.shape[0])
        
        # choose the detections of the (idx) image
        annotations = labels[labels[:,0] == idx][:,1:]
        img_labels = annotations[:,0] if len(annotations) else []
        if len(annotations):
            detected_boxes=[]
            img_boxes = annotations[:,1:]
            
            for pred_idx,(pred_box,pred_label) in enumerate(zip(pred_boxes,pred_cls)):
                
                # if targets are all found, then break
                if len(detected_boxes) == len(annotations):
                    break
                
                # ignore if label is not one of the img labels
                if pred_label not in img_labels:
                    continue
                
                iou,box_idx = bbox_iou(pred_box.unsqueeze(0), img_boxes).max(0)
                if iou > iou_thres and box_idx not in detected_boxes:
                    tp[pred_idx] = 1
                    detected_boxes += [box_idx]
        
        if labels.is_cuda:
            pred_conf = pred_conf.detach().cpu()
            pred_cls = pred_cls.detach().cpu()
        batch_metrics.append([tp,pred_conf,pred_cls])
    return batch_metrics
        
def ap_per_cls(tp,pred_conf,pred_cls,cls_list):
    """
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        pred_conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        cls_list: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # sort by objectness
    idx = np.argsort(-pred_conf)
    tp,pred_conf,pred_cls = tp[idx],pred_conf[idx],pred_cls[idx]
    
    # find unique classes
    unique_cls = np.unique(cls_list)
    print(len(unique_cls))
    
    # create P-R curve and compute AP for each cls
    ap,p,r=[],[],[]
    # c:cls
    for c in tqdm.tqdm(unique_cls,desc="Computing AP"):
        idx = pred_cls ==c
        gt_num = (cls_list == c).sum() # number of ground truth
        p_num = idx.sum() # number of predicted objects
        
        if p_num ==0 and gt_num ==0:
            continue
        elif p_num ==0 or gt_num ==0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # accumulate fps and tps
            fpc = (1-tp[idx]).cumsum()
            tpc = (tp[idx]).cumsum()
            
            # recall 
            recall_curve = tpc/(gt_num+1e-16)
            r.append(recall_curve[-1])
            
            # precision
            precision_curve = tpc/(tpc+fpc)
            p.append(precision_curve[-1])
            
            # ap from recall-precision curve
            ap.append(compute_ap(recall_curve,precision_curve))
            
    # compute f1 score (harmonic mean of precision and recall)
    p,r,ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2*p*r/(p+r+1e-16)
    
    return p,r,ap,f1,unique_cls.astype("int32")

def compute_ap(recall,precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct ap caulculation
    # first append sentinel values at the end
    
    rec = np.concatenate(([0.0],recall,[1.0]))
    pre = np.concatenate(([0.0],precision,[0.0]))
    
    # compute the precision envelope
    for idx in range(pre.size -1, 0, -1):
        pre[idx-1] = np.maximum(pre[idx-1],pre[idx])
    
    # to calculate area under PR curce, look for points
    # where x axis (recall) changes value
    idx = np.where(rec[1:]!=rec[:-1])[0]
    
    # and sum(\delta recall) * prec
    ap = np.sum((rec[idx+1]-rec[idx])*pre[idx+1])
    
    return ap



