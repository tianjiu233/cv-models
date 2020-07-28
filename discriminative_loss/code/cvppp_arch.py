import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ReNet(nn.Module):
    
    def __init__(self, n_input, n_units, patch_size=(1,1)):
        super(ReNet,self).__init__()
        
        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])
        
        assert self.patch_size_height>=1
        assert self.patch_size_width>=1
        
        if ((self.patch_size_height==1) and (self.patch_size_width==1)):
            self.tiling = False
        else:
            self.tiling = True
            
        self.rnn_hor = nn.GRU(input_size = n_input*self.patch_size_height*self.patch_size_width,
                              hidden_size = n_units, 
                              num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_ver = nn.GRU(input_size = n_units*2,
                              hidden_size = n_units,
                              num_layers=1,batch_first=True,bidirectional=True)
    
    def tile(self,x):
        n_height_padding = self.patch_size_height - x.size(2)%self.patch_size_height
        n_width_padding = self.patch_size_width - x.size(3)%self.patch_size_width
        
        n_top_padding = int(n_height_padding/2)
        n_bottom_padding = int(n_height_padding - n_top_padding)
        
        n_left_padding = int(n_width_padding/2)
        n_right_padding = int(n_width_padding - n_left_padding)
        
        x = F.pad(x,(n_left_padding,n_right_padding,n_top_padding,n_bottom_padding))
        
        b, n_filters, n_height, n_width = x.size()
        
        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0
        
        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width
        
        x = x.view(b, n_filters, new_height, self.patch_size_height, new_width, self.patch_size_width)
        x = x.permute(0,2,4,1,3,5)
        x = x.contiguous()
        x = x.view(b,new_height,new_width,self.patch_size_height*self.patch_size_width*n_filters)
        x = x.permute(0,3,1,2)
        x = x.contiguous()
        
        return x
    
    def rnn_forward(self,x,hor_or_ver):
        assert hor_or_ver in ["hor","ver"]
        
        b, n_height, n_width, n_filters = x.size()
        
        x = x.view(b*n_height, n_width, n_filters)
        
        if hor_or_ver == "hor":
            x, _ = self.rnn_hor(x)
        else:
            x, _ = self.rnn_ver(x)
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)
        
        return x
    
    def forward(self,x):
        
        if self.tiling:
            x = self.tile(x) # b,nf,h,w
        
        x = x.permute(0,2,3,1) # b,h,w,nf
        x= x.contiguous()
        x = self.rnn_forward(x,"hor") # b,h,w,nf
        x = x.permute(0,2,1,3) # b,w,h,nf
        x = x.contiguous()
        x = self.rnn_forward(x,"ver") # b,w,h,nf
        x = x.permute(0,2,1,3) # b,h,w,nf
        x = x.contiguous()
        x = x.permute(0,3,1,2)
        x = x.contiguous()
        
        return x

class ModifiedVGG(nn.Module):
    def __init__(self):
        super(ModifiedVGG, self).__init__()
        self.outputs = [3,8]
        self.n_filters = [64,128]
        
        self.model = models.__dict__["vgg16"](pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[0])
        self.model = nn.Sequential(*list(self.model.children())[:16])
    def forward(self,x):
        out = []
        for i, layer in enumerate(self.model.children()):
            x = layer(x)
            if i in self.outputs:
                out.append(x)
        out.append(x)
        
        return out

class BaseCNN(nn.Module):
    def __init__(self,use_coords=False):
        super(BaseCNN, self).__init__()
        
        self.model = ModifiedVGG()
        self.n_filters = self.model.n_filters
        
        if use_coords:
            first_layer = list(list(self.model.children())[0].modules())[1]
            addtional_weights = torch.zeros(64,2,3,3)
            new_weights = torch.cat(
                    (addtional_weights, first_layer.weight.data),dim=1)
            new_weights = torch.nn.Parameter(new_weights)
            
            first_layer.weight = new_weights
            first_layer.in_channels = 5
    def forward(self,x):
        b,n_channels, n_height, n_width = x.size()
        out = self.model(x)
        return out
    
class Architecture(nn.Module):
    def __init__(self,n_classes, use_instance_seg, use_coords, pixel_embedding_dim=16):
        super(Architecture, self).__init__()
        
        self.n_classes = n_classes
        self.use_instance_seg = use_instance_seg
        self.use_coords = use_coords
        
        self.cnn = BaseCNN(use_coords = self.use_coords)
        self.renet1 = ReNet(n_input=256,n_units=100)
        self.renet2 = ReNet(n_input=100*2,n_units=100)
        
        self.upsampling1 = nn.ConvTranspose2d(
                in_channels=100*2,
                out_channels=100, 
                kernel_size=(2,2),
                stride=(2,2),)
        self.relu1 = nn.ReLU()
        self.upsampling2 = nn.ConvTranspose2d(
                in_channels=100 + self.cnn.n_filters[1],
                out_channels=100,
                kernel_size=(2,2),
                stride=(2,2),)
        self.relu2 = nn.ReLU()
        
        self.sem_seg_output = nn.Conv2d(
                in_channels = 100+self.cnn.n_filters[0],
                out_channels = self.n_classes,
                kernel_size=(1,1),
                stride=(1,1))
        
        if self.use_instance_seg:
            self.ins_seg_output = nn.Conv2d(
                    in_channels = 100+self.cnn.n_filters[0],
                    out_channels = pixel_embedding_dim,
                    kernel_size=(1,1),
                    stride=(1,1))
        
        self.ins_cls_cnn = nn.Sequential()
        self.ins_cls_cnn.add_module("pool1", nn.MaxPool2d(2,stride=2))
        self.ins_cls_cnn.add_module("conv1", nn.Conv2d(
                in_channels=100*2,out_channels=64,
                kernel_size=(3,3),stride=(1,1),padding=(1,1)))
        self.ins_cls_cnn.add_module("relu1",nn.ReLU())
        self.ins_cls_cnn.add_module("conv2", nn.Conv2d(
                in_channels=64,out_channels=64,
                kernel_size=(3,3),stride=(1,1),padding=(1,1)))
        self.ins_cls_cnn.add_module("relu2",nn.ReLU())
        self.ins_cls_cnn.add_module("pool2",nn.MaxPool2d(kernel_size=2,stride=2))
        self.ins_cls_cnn.add_module("conv3", nn.Conv2d(
                in_channels=64,out_channels=64,
                kernel_size=(3,3),stride=(1,1),padding=(1,1)))
        self.ins_cls_cnn.add_module("relu3",nn.ReLU())
        self.ins_cls_cnn.add_module("conv4", nn.Conv2d(
                in_channels=64,out_channels=64,
                kernel_size=(3,3),stride=(1,1),padding=(1,1)))
        self.ins_cls_cnn.add_module("relu4",nn.ReLU())
        self.ins_cls_cnn.add_module("pool3",nn.AdaptiveAvgPool2d((1,1))) # b, nf, 1, 1
        
        self.ins_cls_out = nn.Sequential()
        self.ins_cls_out.add_module("linear",nn.Linear(64,1))
        self.ins_cls_out.add_module("sigmoid",nn.Sigmoid())
        
    def forward(self,x):
        
        first_skip, second_skip, x_enc = self.cnn(x)
        
        x_enc = self.renet1(x_enc)
        x_enc = self.renet2(x_enc)
        x_dec = self.relu1(self.upsampling1(x_enc))
        x_dec = torch.cat((x_dec, second_skip),dim=1)
        x_dec = self.relu2(self.upsampling2(x_dec))
        x_dec = torch.cat((x_dec, first_skip),dim=1)
        
        sem_seg_out = self.sem_seg_output(x_dec)
        
        if self.use_instance_seg:
            ins_seg_out = self.ins_seg_output(x_dec)
        else:
            ins_seg_out = None
        
        x_ins_cls = self.ins_cls_cnn(x_enc)
        x_ins_cls = x_ins_cls.squeeze(3).squeeze(2)
        x_ins_cls = self.ins_cls_out(x_ins_cls)
        
        return sem_seg_out, ins_seg_out, x_ins_cls
        
        
    
if __name__=="__main__":
    
    renet = ReNet(n_input=3,n_units=100,patch_size=(32,32))
    input_sample = torch.randn(2,3,256,256)
    
    net = Architecture(n_classes=2, use_instance_seg=True, use_coords=False)
    sem,ins,num = net(input_sample)
    """
    sem:(bs,n_class,height,width)
    ins:(bs,n_filters,height,width)
    num:(bs,1)
    """