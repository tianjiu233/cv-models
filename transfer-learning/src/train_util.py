# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:27:54 2020

@author: huijianpzh
"""

# the file includes model init, lr scheduler.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear")!=-1:
        return
    elif classname.find("")