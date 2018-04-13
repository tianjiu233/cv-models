import torchvision.transforms as transforms
import torch.utils as utils 
from skimage import io as skio
from skimage import transform as sktrans
import matplotlib.pyplot as plt
import numpy as np
import os

class Nptranspose(object):
    def __call__(self,sample):
        """
        This class is just set to do the transpose operation.
        It should be added in every compose.
        """
        image = sample["image"]
        label = sample["label"]
        image = image.transpose(2,0,1)
        label = label.transpose(2,0,1)
        
        return {"image":image, "label":label}
        

class RandomCrop(object):
    """Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired ouput size, if int, square crop 
        is made.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
            
    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        h,w = image.shape[0:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        
        image = image[top:top+new_h, left:left+new_w,:]
        label = label[top:top+new_h,left:left+new_w,:]
        return {'image':image, 'label':label}

class Jitter(object):
    def __init__(self,size=(286,286),crop_size=256,p=0.5):
        self.crop_size=256
        self.size = size
        self.p=p
    def __call__(self,sample):
        
        image, label = sample['image'], sample['label']
        
        if np.random.random()>self.p:
            return {'image':image, 'label':label}
        else:
            image = sktrans.resize(image,self.size,mode="reflect").astype(np.float32)
            label = sktrans.resize(label,self.size,mode="reflect").astype(np.float32)
            
            return RandomCrop(self.crop_size)({'image':image, 'label':label})            

class H_Mirror(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            new_image = np.flip(image,0).copy()
            new_label = np.flip(label,0).copy()
            return {'image':new_image, 'label':new_label}
        else:
            return {'image':image, 'label':label}

class V_Mirror(object):
    def __init__(self,p=0.5):
        self.p=p
    def __call__(self,sample):
        image, label = sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            new_image = np.flip(image,1).copy()
            new_label = np.flip(label,1).copy()
            return {'image':new_image, 'label':new_label}
        else:
            return {'image':image, 'label':label}
            

class ABDataset(utils.data.Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.train_data = []
        files = os.listdir(self.root_dir)
        for item in files:
            if item.endswith(".jpg"):
                self.train_data.append(item.split(".jpg")[0])
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        
        prefix = self.root_dir
        img_name = prefix+self.train_data[index]+".jpg"
        image = skio.imread(img_name)
        image = image.astype(np.float32)/(255*0.5) - 1
        
        row,col,channel = image.shape
        
        # AtoB
        new_image = image[:,0:int(col/2),:]
        image_label = image[:,int(col/2):,:]
        
        sample = {}
        sample['image'] = new_image
        sample['label'] = image_label
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

if __name__ == "__main__":
    print("dataio.py testing...")
    composed = transforms.Compose([RandomCrop(256),Jitter(),H_Mirror(),V_Mirror(),Nptranspose()])
    maps_dataset = ABDataset(root_dir = "../maps/train/",transform = composed)
    
    idx = np.random.randint(0,len(maps_dataset))
    sample = maps_dataset[idx]
    # to show, change the axis
    image = (sample["image"].transpose(1,2,0) + 1)*(255*0.5)
    label = (sample["label"].transpose(1,2,0) + 1)*(255*0.5)
    
    if True:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(image.astype(np.uint8))
        ax[1].imshow(label.astype(np.uint8))
        ax[0].set_title("Image")
        ax[1].set_title("Label")
        plt.show()
    
    loader = utils.data.DataLoader(maps_dataset,batch_size=4)