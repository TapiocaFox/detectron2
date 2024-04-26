from math import sqrt
from itertools import product as product
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from insa import *
from utils.utils import *

warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class INSAFusion(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self):
        super(VGGBase, self).__init__()

        # RGB
        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)        
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)
        
        # LWIR
        self.conv1_1_lwir = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_lwir = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_lwir = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_lwir = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_lwir = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_lwir = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_lwir = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_lwir = nn.BatchNorm2d(128, affine=True)

        self.pool2_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_lwir = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_lwir = nn.BatchNorm2d(256, affine=True)
        
        # Final pooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        # INSA fusion
        self.conv1x1_vis = nn.Conv2d(256,256,kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1x1_vis.weight.data.normal_(0, 0.01)
        self.conv1x1_vis.bias.data.fill_(0.01)

        self.conv1x1_lwir = nn.Conv2d(256,256,kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1x1_lwir.weight.data.normal_(0, 0.01)
        self.conv1x1_lwir.bias.data.fill_(0.01)   
        
        # INSA fusion weight between rbg and lwir
        self.weight = 0.5
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  
        nn.init.constant_(self.rescale_factors, 20)

        # Load pretrained layers
        self.load_pretrained_layers()

        # INtra-INter Attention (INSA) module
        self.insa = INSA(n_iter=2,
                         dim=256,
                         n_head=1,
                         ffn_dim=4)
        

    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        # RGB
        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis)))  
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis))) 
        out_vis = self.pool1_vis(out_vis)  

        out_vis = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))
        out_vis = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis))) 
        out_vis = self.pool2_vis(out_vis) 

        out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis))) 
        out_vis = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis))) 
        out_vis = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis)))

        # LWIR
        out_lwir = F.relu(self.conv1_1_bn_lwir(self.conv1_1_lwir(image_lwir)))  
        out_lwir = F.relu(self.conv1_2_bn_lwir(self.conv1_2_lwir(out_lwir))) 
        out_lwir = self.pool1_lwir(out_lwir)  

        out_lwir = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir)))
        out_lwir = F.relu(self.conv2_2_bn_lwir(self.conv2_2_lwir(out_lwir))) 
        out_lwir = self.pool2_lwir(out_lwir) 

        out_lwir = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir))) 
        out_lwir = F.relu(self.conv3_2_bn_lwir(self.conv3_2_lwir(out_lwir))) 
        out_lwir = F.relu(self.conv3_3_bn_lwir(self.conv3_3_lwir(out_lwir))) 
        
        # INSA fusion
        out_vis = F.relu(self.conv1x1_vis(out_vis))
        out_lwir = F.relu(self.conv1x1_lwir(out_lwir))

        out_vis, out_lwir = self.insa(out_vis, out_lwir)
        
        # Weighted summation
        out = torch.add(out_vis * self.weight, out_lwir * (1 - self.weight))
        
        # Final pooling
        out = self.pool3(out)
        return out


    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of model
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG_BN
        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[1:50]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[50:99]):    
            if param == 'conv1_1_lwir.weight':
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]][:, :1, :, :]              
            else:
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        self.load_state_dict(state_dict)

        print("Load Model: INSANet\n")

