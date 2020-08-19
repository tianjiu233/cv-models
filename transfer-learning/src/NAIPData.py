# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:45:39 2020

@author: huijianpzh
"""

# official libs
import torchvision
import torch
from torch.utils.data import Dataset,DataLoader

import os
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt

from skimage import io

# my libs
from data_util import Nptranspose,H_Mirror,V_Mirror,Rotation,ColorAug,RandomCrop

def hex2rgb(hex_string):
    hex_string = hex_string.lstrip("#")
    return tuple(int(hex_string[i:i+2],16)/256.0 for i in (0,2,4))


NLCD_CLASSES  = collections.OrderedDict([
    (11, ("Open Water", "#5475A8", "areas of open water, generally with less than 25% cover of vegetation or soil.")),
    (12, ("Perennial Ice/Snow", "#FFFFFF", "areas characterized by a perennial cover of ice and/or snow, generally greater than 25% of total cover.")),
    (21, ("Developed, Open Space", "#E8D1D1", "areas with a mixture of some constructed materials, but mostly vegetation in the form of lawn grasses. Impervious surfaces account for less than 20% of total cover. These areas most commonly include large-lot single-family housing units, parks, golf courses, and vegetation planted in developed settings for recreation, erosion control, or aesthetic purposes.")),
    (22, ("Developed, Low Intensity", "#E29E8C", "areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 20% to 49% percent of total cover. These areas most commonly include single-family housing units.")),
    (23, ("Developed, Medium Intensity", "#FF0000", "areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 50% to 79% of the total cover. These areas most commonly include single-family housing units.")),
    (24, ("Developed High Intensity", "#B50000", "highly developed areas where people reside or work in high numbers. Examples include apartment complexes, row houses and commercial/industrial. Impervious surfaces account for 80% to 100% of the total cover.")),
    (31, ("Barren Land (Rock/Sand/Clay)", "#D2CDC0", "areas of bedrock, desert pavement, scarps, talus, slides, volcanic material, glacial debris, sand dunes, strip mines, gravel pits and other accumulations of earthen material. Generally, vegetation accounts for less than 15% of total cover.")),
    (41, ("Deciduous Forest", "#85C77E", "areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species shed foliage simultaneously in response to seasonal change.")),
    (42, ("Evergreen Forest", "#38814E", "areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species maintain their leaves all year. Canopy is never without green foliage.")),
    (43, ("Mixed Forest", "#D4E7B0", "areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. Neither deciduous nor evergreen species are greater than 75% of total tree cover.")),
    (52, ("Shrub/Scrub", "#DCCA8F", "areas dominated by shrubs; less than 5 meters tall with shrub canopy typically greater than 20% of total vegetation. This class includes true shrubs, young trees in an early successional stage or trees stunted from environmental conditions.")),
    (71, ("Grassland/Herbaceous", "#FDE9AA", "areas dominated by gramanoid or herbaceous vegetation, generally greater than 80% of total vegetation. These areas are not subject to intensive management such as tilling, but can be utilized for grazing.")),
    (81, ("Pasture/Hay", "#FBF65D", "areas of grasses, legumes, or grass-legume mixtures planted for livestock grazing or the production of seed or hay crops, typically on a perennial cycle. Pasture/hay vegetation accounts for greater than 20% of total vegetation.")),
    (82, ("Cultivated Crops", "#CA9146", "areas used for the production of annual crops, such as corn, soybeans, vegetables, tobacco, and cotton, and also perennial woody crops such as orchards and vineyards. Crop vegetation accounts for greater than 20% of total vegetation. This class also includes all land being actively tilled.")),
    (90, ("Woody Wetlands", "#C8E6F8", "areas where forest or shrubland vegetation accounts for greater than 20% of vegetative cover and the soil or substrate is periodically saturated with or covered with water.")),
    (95, ("Emergent Herbaceous Wetlands", "#64B3D5", "Areas where perennial herbaceous vegetation accounts for greater than 80% of vegetative cover and the soil or substrate is periodically saturated with or covered with water."))
])

HR_CLASSES = collections.OrderedDict([
    (1, ("Water", "#0000FF", "")),
    (2, ("Forest", "#007F00", "")),
    (3, ("Field", "#7FFF7F", "")),
    (4, ("Impervious Surfaces", "#7F6060", ""))
])

# Acutally, they are not used. In the official codes, the number of categories is 22 or 21
HR_LU_CLASSES = collections.OrderedDict([
    (1, ("Impervious, Road", "#000000", "")),
    (2, ("Impervious, Non-Road", "#730e04", "")),
    (3, ("Tree Canopy over Impervious Surfaces", "#7df707", "")),
    (4, ("Water", "#1870ff", "")),
    (5, ("Tidal Wetlands", "#7afac5", "")),
    (6, ("Floodplain Wetlands", "#70e6a9", "")),
    (7, ("Other Wetlands", "#70e6a9", "")),
    (8, ("Forest", "#377301", "")),
    (9, ("Tree Canopy over Turf Grass", "#a7f804", "")),
    (10, ("Mixed Open", "#a86f00", "")),
    (11, ("Fractional Turf (small)", "#f3bde8", "")),
    (12, ("Fractional Turf (med)", "#f3bde8", "")),
    (13, ("Fractional Turf (large)", "#f3bde8", "")),
    (14, ("Fractional Impervious", "#c746ff", "")),
    (15, ("Turf Grass", "#fefb73", "")),
    (16, ("Cropland", "#e69800", "")),
    (17, ("Pasture/Hay", "#e69800", ""))
])

HR_CLASSES_CMAP = matplotlib.colors.ListedColormap([
    HR_CLASSES[i][1] if i in HR_CLASSES else "#000000"
    for i in range(0, max(HR_CLASSES)+1)
])

NLCD_CLASSES_CMAP = matplotlib.colors.ListedColormap([
    NLCD_CLASSES[i][1] if i in NLCD_CLASSES else "#000000"
    for i in range(0, max(NLCD_CLASSES)+1)
])

# the special methods for NAIP data
class PrepareData(object):
    """
    It can be also viewed as the Label Overloading method.
    """
    def __init__(self,p=1):
        self.p = p
    def __call__(self,sample):
        
        data_naip_old = sample["data_naip_old"]
        data_naip_new = sample["data_naip_new"]
        
        data_lc = sample["data_lc"]
        data_nlcd = sample["data_nlcd"]
        
        new_sample = {}
        
        if np.random.random()<=self.p:
            new_sample["image"] = data_naip_new / 255.
        else:
            new_sample["image"] = data_naip_old /255.
        
        # label_lc,label_nlcd from 2d->3d [h,w] -> [h,w,1]
        data_lc = np.expand_dims(data_lc,axis=2)
        data_nlcd = np.expand_dims(data_nlcd,axis=2)
        
        # new_sample["label_lc"] = data_lc
        # new_sample["label_nlcd"] = data_nlcd
        
        "we only keep hr resolution label here"
        new_sample["label"] = data_lc
        
        return new_sample

class NAIPData(Dataset):
    def __init__(self,data_dir,
                 nlcd_key_txt,
                 transform=None,
                 mode="npz",nir=False):
        
        self.nir = nir
        self.mode = mode
        self.data_dir = data_dir
        self.data = []
        
        self.transform = transform
        
        if mode == "npz":
            self.suffix = ".npz"
            files = os.listdir(data_dir+"/")
            for item in files:
                if item.endswith(self.suffix):
                    self.data.append(item.split(self.suffix)[0])
        else:
            self.suffix = ".tif"
            files = os.listdir(data_dir+"/")
            for item in files:
                if "naip-new" in item:
                    self.data.append(item.split("naip-new")[0])
        
        # do some basic ops here
        self.key_array = np.loadtxt(nlcd_key_txt)
        
    def __len__(self):
        return len(self.data)
    
    
    def nlcd2lr(self,trans_arr):
        # nlcd->lr
        for translation in self.key_array:
            # translation is (src label, dst label)
            scr_l,dst_l = translation
            if scr_l != dst_l:
                trans_arr[trans_arr==scr_l] = dst_l
        return trans_arr
    
    def lr2nlcd(self,trans_arr):
        for translation in self.key_array:
            # translation is (dst label, src label)
            # this is the only difference
            dst_l,scr_l = translation
            if scr_l != dst_l:
                trans_arr[trans_arr==scr_l] = dst_l
        return trans_arr
    
    def __getitem__(self, index):
        
        if self.suffix ==".npz":
            fn = self.data_dir + "/" + self.data[index] + self.suffix 
            with np.load(fn) as f:
                data = f["arr_0"].squeeze()
                data = np.rollaxis(data,0,3)
                
                # image  R,G,B&NIR
                if self.nir:
                    data_naip_new = data[:,:,0:4]
                    data_naip_old = data[:,:,4:8]
                else:
                    data_naip_new = data[:,:,0:3]
                    data_naip_old = data[:,:,4:7]
            
                # label
                # high resolution label
                # [0,1,2,3,4]->[0,1,2,3,4]
                data_lc = data[:,:,8]
                # imprevious will be taken into barren land
                data_lc[data_lc==5] = 4
                data_lc[data_lc==6] = 4
                # no data to 0
                data_lc[data_lc==15] = 0
                # low resolution 
                data_nlcd = data[:,:,9]
                # nlcd2lr
                data_nlcd = self.nlcd2lr(data_nlcd)
            
            
        else:
            # suffix: ".tif"
            data_naip_new = self.data_dir + "/" + self.data[index] + "naip-new" + self.suffix
            data_naip_old = self.data_dir + "/" + self.data[index] + "naip-old" + self.suffix
            data_lc = self.data_dir + "/" + self.data[index] + "lc" + self.suffix
            data_nlcd = self.data_dir + "/" + self.data[index] + "nlcd" + self.suffix
            
            
            data_naip_new = io.imread(data_naip_new)
            data_naip_old = io.imread(data_naip_old)
            
            if self.nir == False:
                data_naip_new = data_naip_new[:,:,0:3]
                data_naip_old = data_naip_old[:,:,0:3]
            
            data_lc = io.imread(data_lc)
            data_lc[data_lc==5] = 4
            data_lc[data_lc==6] = 4
            data_nlcd = io.imread(data_nlcd)
            
        sample = {}
        sample["data_naip_new"] = data_naip_new
        sample["data_naip_old"] = data_naip_old
        sample["data_lc"] = data_lc.astype(np.float32)
        sample["data_nlcd"] = data_nlcd.astype(np.float32)   
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def show_patch(self, index):
        
        sample = self.__getitem__(index)
        
        data_naip_new = sample["data_naip_new"]
        data_naip_old = sample["data_naip_old"]
        
        data_lc = sample["data_lc"]
        data_nlcd = sample["data_nlcd"]
        data_nlcd = self.lr2nlcd(data_nlcd)
        
        #print(data_naip_new.shape)
        fig,axs = plt.subplots(1,4,figsize=(14,4.6))

        axs[0].imshow(data_naip_new.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(data_naip_old.astype(np.uint8))
        axs[1].axis("off")
        axs[2].imshow(data_nlcd,cmap=NLCD_CLASSES_CMAP,vmin=0,vmax=NLCD_CLASSES_CMAP.N)
        axs[2].axis("off")
        axs[3].imshow(data_lc,cmap=HR_CLASSES_CMAP,vmin=0,vmax=HR_CLASSES_CMAP.N)
        axs[3].axis("off")
        plt.suptitle(self.data[index],y=0.94)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    def show_sample(self,index):
        """
        This function is a liittle different for that we should change the axis and  then we can get the image.
        """
        sample = self.__getitem__(index)
        
        image,label_lc = sample["image"],sample["label"]
        
        #print(image.shape)
        
        image = image.transpose(1,2,0)*255
        image = image.astype(np.uint8)
        label_lc = label_lc.transpose(1,2,0)
        label_lc = label_lc[:,:,0]
        
        fig,axs = plt.subplots(1,2)
        
        # print(image.shape)
        
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(label_lc,cmap=HR_CLASSES_CMAP,vmin=0,vmax=HR_CLASSES_CMAP.N)
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()



class NAIPDataList(Dataset):
    def __init__(self,data_dir_list,
                 nlcd_key_txt,
                 transform=None,
                 mode="npz",nir=False):
        
        self.nir = nir
        self.mode = mode
        assert len(data_dir_list)>0
        self.data_dir_list = data_dir_list
        self.data = []
        
        self.transform = transform
        
        if mode == "npz":
            self.suffix = ".npz"
        else:
            self.suffix = ".tif"
            
        for idx in range(len(self.data_dir_list)):
            
            data_dir = self.data_dir_list[idx]
            
            files = os.listdir(data_dir+"/")
            
            for item in files:
                if mode == "npz":
                    if item.endswith(self.suffix):
                        self.data.append(data_dir + "/" + item.split(self.suffix)[0])
                else:
                    if "naip-new" in item:
                        self.data.append(data_dir + "/" + item.split("naip-new")[0])
        
        # do some basic ops here
        self.key_array = np.loadtxt(nlcd_key_txt)
        
    def __len__(self):
        return len(self.data)
    
    
    def nlcd2lr(self,trans_arr):
        # nlcd->lr
        for translation in self.key_array:
            # translation is (src label, dst label)
            scr_l,dst_l = translation
            if scr_l != dst_l:
                trans_arr[trans_arr==scr_l] = dst_l
        return trans_arr
    
    def lr2nlcd(self,trans_arr):
        for translation in self.key_array:
            # translation is (dst label, src label)
            # this is the only difference
            dst_l,scr_l = translation
            if scr_l != dst_l:
                trans_arr[trans_arr==scr_l] = dst_l
        return trans_arr
    
    def __getitem__(self, index):
        
        if self.suffix ==".npz":
            fn = self.data[index] + self.suffix
            with np.load(fn) as f:
                data = f["arr_0"].squeeze()
                data = np.rollaxis(data,0,3)
                
                # image  R,G,B&NIR
                if self.nir:
                    data_naip_new = data[:,:,0:4]
                    data_naip_old = data[:,:,4:8]
                else:
                    data_naip_new = data[:,:,0:3]
                    data_naip_old = data[:,:,4:7]
            
                # label
                # high resolution label
                # [0,1,2,3,4]->[0,1,2,3,4]
                data_lc = data[:,:,8]
                # imprevious will be taken into barren land
                data_lc[data_lc==5] = 4
                data_lc[data_lc==6] = 4
                # no data to 0
                data_lc[data_lc==15] = 0
                # low resolution 
                data_nlcd = data[:,:,9]
                # nlcd2lr
                data_nlcd = self.nlcd2lr(data_nlcd)
            
            
        else:
            # suffix: ".tif"
            data_naip_new = self.data[index] + "naip-new" + self.suffix
            data_naip_old = self.data[index] + "naip-old" + self.suffix
            data_lc = self.data[index] + "lc" + self.suffix
            data_nlcd = self.data[index] + "nlcd" + self.suffix
            
            data_naip_new = io.imread(data_naip_new)
            data_naip_old = io.imread(data_naip_old)
            
            if self.nir == False:
                data_naip_new = data_naip_new[:,:,0:3]
                data_naip_old = data_naip_old[:,:,0:3]
            
            data_lc = io.imread(data_lc)
            data_lc[data_lc==5] = 4
            data_lc[data_lc==6] = 4
            data_nlcd = io.imread(data_nlcd)
            
        sample = {}
        sample["data_naip_new"] = data_naip_new
        sample["data_naip_old"] = data_naip_old
        sample["data_lc"] = data_lc.astype(np.float32)
        sample["data_nlcd"] = data_nlcd.astype(np.float32)   
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def show_patch(self, index):
        
        sample = self.__getitem__(index)
        
        data_naip_new = sample["data_naip_new"]
        data_naip_old = sample["data_naip_old"]
        
        data_lc = sample["data_lc"]
        data_nlcd = sample["data_nlcd"]
        data_nlcd = self.lr2nlcd(data_nlcd)
        
        #print(data_naip_new.shape)
        fig,axs = plt.subplots(1,4,figsize=(14,4.6))

        axs[0].imshow(data_naip_new.astype(np.uint8))
        axs[0].axis("off")
        axs[1].imshow(data_naip_old.astype(np.uint8))
        axs[1].axis("off")
        axs[2].imshow(data_nlcd,cmap=NLCD_CLASSES_CMAP,vmin=0,vmax=NLCD_CLASSES_CMAP.N)
        axs[2].axis("off")
        axs[3].imshow(data_lc,cmap=HR_CLASSES_CMAP,vmin=0,vmax=HR_CLASSES_CMAP.N)
        axs[3].axis("off")
        plt.suptitle(self.data[index],y=0.94)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    def show_sample(self,index):
        """
        This function is a liittle different for that we should change the axis and  then we can get the image.
        """
        sample = self.__getitem__(index)
        
        image,label_lc = sample["image"],sample["label"]
        
        #print(image.shape)
        
        image = image.transpose(1,2,0)*255
        image = image.astype(np.uint8)
        label_lc = label_lc.transpose(1,2,0)
        label_lc = label_lc[:,:,0]
        
        fig,axs = plt.subplots(1,2)
        
        # print(image.shape)
        
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(label_lc,cmap=HR_CLASSES_CMAP,vmin=0,vmax=HR_CLASSES_CMAP.N)
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()





if __name__== "__main__":
    nlcd_key_txt = "nlcd_to_lr_labels.txt"
    
    """
    # ------ NAIP part ------
    # test 4 .npz suffix
    npz_dir = r"D:\repo\data\de_1m_2013\de_1m_2013_extended-val_patches"
    data_transform = torchvision.transforms.Compose([PrepareData(0.5),
                                                     H_Mirror(),V_Mirror(),
                                                     Rotation(),ColorAug(),Nptranspose()])
    npzdata = NAIPData(npz_dir,
                       nlcd_key_txt,
                       transform=data_transform,
                       mode="npz",
                       nir=False)
    index = 5
    if data_transform is None:
        npzdata.show_patch(index)
    else:
        npzdata.show_sample(index)
        
        
    # test 4 .tif suffix
    tif_dir = r"D:\repo\data\de_1m_2013\de_1m_2013_extended-train_tiles"
    data_transform = torchvision.transforms.Compose([PrepareData(0.5),
                                                     RandomCrop(512),
                                                     H_Mirror(),V_Mirror(),
                                                     Rotation(),ColorAug(),Nptranspose()])
    tifdata = NAIPData(tif_dir,
                       nlcd_key_txt,
                       transform=data_transform,
                       mode="tif",
                       nir=False)
    index = 10
    if data_transform is None:
        tifdata.show_patch(index)
    else:
        tifdata.show_sample(index)
    """
    
    
    # ------ NAIPDATAList Part ------
    # test 4 npz data
    npz_data_dir_list= [r"D:\repo\data\de_1m_2013\de_1m_2013_extended-val_patches",
                        r"D:\repo\data\ny_1m_2013\ny_1m_2013_extended-train_patches",
                        ]
    mode = "npz"
    data_transform = torchvision.transforms.Compose([PrepareData(0.5),
                                                     H_Mirror(),V_Mirror(),
                                                     Rotation(),ColorAug(),Nptranspose()])
    npzdata = NAIPDataList(data_dir_list=npz_data_dir_list,
                           nlcd_key_txt=nlcd_key_txt,
                           transform=data_transform,
                           mode=mode,nir=False)
    index = 100
    if data_transform is None:
        npzdata.show_patch(index)
    else:
        npzdata.show_sample(index)
        
    # test 4 tif data
    tif_data_dir_list= [r"D:\repo\data\de_1m_2013\de_1m_2013_extended-train_tiles",
                        r"D:\repo\data\ny_1m_2013\ny_1m_2013_extended-train_tiles",
                        ]
    mode = "tif"
    data_transform = torchvision.transforms.Compose([PrepareData(0.5),
                                                     H_Mirror(),V_Mirror(),
                                                     Rotation(),ColorAug(),Nptranspose()])
    tifdata = NAIPDataList(data_dir_list=tif_data_dir_list,
                           nlcd_key_txt=nlcd_key_txt,
                           transform=data_transform,
                           mode=mode,nir=False)
    index = 10
    if data_transform is None:
        tifdata.show_patch(index)
    else:
        tifdata.show_sample(index)
    
    
    