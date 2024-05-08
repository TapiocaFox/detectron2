from math import sqrt
from itertools import product as product
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

from insa import *
# from utils.utils import *

warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@BACKBONE_REGISTRY.register()
class INSAFusion(Backbone):
    def __init__(self, cfg, input_shape=None):
        super(INSAFusion, self).__init__()

        # RGB pathways
        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_1_bn_vis = nn.BatchNorm2d(64)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64)
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128)
        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256)
        self.pool3_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # LWIR pathways
        self.conv1_1_lwir = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_1_bn_lwir = nn.BatchNorm2d(64)
        self.conv1_2_lwir = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_lwir = nn.BatchNorm2d(64)
        self.pool1_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_lwir = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_lwir = nn.BatchNorm2d(128)
        self.conv2_2_lwir = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_lwir = nn.BatchNorm2d(128)
        self.pool2_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_lwir = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_lwir = nn.BatchNorm2d(256)
        self.conv3_2_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_lwir = nn.BatchNorm2d(256)
        self.conv3_3_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_lwir = nn.BatchNorm2d(256)
        self.pool3_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # INSA fusion
        self.insa = INSA(n_iter=2, dim=256, n_head=1, ffn_dim=4)
        
        # Weight parameter for fusion
        self.weight = 0.5

        if cfg.MODEL.BACKBONE.PRETRAINED:
            self.load_pretrained_layers()

    def forward(self, x):
        image_vis, image_lwir = x[:, 0:3, :, :], x[:, 3:4, :, :]

        # Process Visible pathway
        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis)))
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis)))
        p2 = self.pool1_vis(out_vis)

        out_vis = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(p2)))
        out_vis = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis)))
        p3 = self.pool2_vis(out_vis)

        out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(p3)))
        out_vis = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis)))
        out_vis = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis)))
        p4 = self.pool3_vis(out_vis)

        # Process LWIR pathway
        out_lwir = F.relu(self.conv1_1_bn_lwir(self.conv1_1_lwir(image_lwir)))
        out_lwir = F.relu(self.conv1_2_bn_lwir(self.conv1_2_lwir(out_lwir)))
        out_lwir = self.pool1_lwir(out_lwir)

        out_lwir = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir)))
        out_lwir = F.relu(self.conv2_2_bn_lwir(self.conv2_2_lwir(out_lwir)))
        out_lwir = self.pool2_lwir(out_lwir)

        out_lwir = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir)))
        out_lwir = F.relu(self.conv3_2_bn_lwir(self.conv3_2_lwir(out_lwir)))
        out_lwir = F.relu(self.conv3_3_bn_lwir(self.conv3_3_lwir(out_lwir)))
        out_lwir = self.pool3_lwir(out_lwir)

        # Fusion
        out_vis, out_lwir = self.insa(out_vis, out_lwir)
        fused_out = torch.add(out_vis * self.weight, out_lwir * (1 - self.weight))

        p5 = self.pool3_vis(fused_out)
        p6 = self.pool3_vis(p5)

        return {
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6
        }

    def load_pretrained_layers(self):
        """
        Load pretrained VGG model weights into the convolutional layers.
        Modify the first convolutional layer for the LWIR input.
        """
        pretrained_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        model_dict = self.state_dict()
        
        # Update the model dict with pretrained weights
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                model_dict[k] = v
            elif 'conv1_1_lwir.weight' in k:
                model_dict[k] = v.mean(dim=1, keepdim=True)
        self.load_state_dict(model_dict)

        print("Loaded pretrained weights from VGG-16.")

    def output_shape(self):
        return {
            'p2': ShapeSpec(channels=64, stride=4),
            'p3': ShapeSpec(channels=128, stride=8),
            'p4': ShapeSpec(channels=256, stride=16),
            'p5': ShapeSpec(channels=256, stride=32),
            'p6': ShapeSpec(channels=256, stride=64),
        }

@BACKBONE_REGISTRY.register()
class INSAFusionBackup(Backbone):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self, cfg, input_shape=None):
        super(INSAFusion, self).__init__()

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

        pretrained = cfg.MODEL.BACKBONE.PRETRAINED if hasattr(cfg.MODEL.BACKBONE, 'PRETRAINED') else False
        if pretrained:
            self.load_pretrained_layers()


    def forward(self, x):
        # print("forward")
        image_vis, image_lwir = x[:, 0:3, :, :], x[:, 3:4, :, :]
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
        # print("CHECKPOINT CHECKPOINT CHECKPOINT CHECKPOINT CHECKPOINT")
        # INSA fusion
        out_vis = F.relu(self.conv1x1_vis(out_vis))
        out_lwir = F.relu(self.conv1x1_lwir(out_lwir))

        out_vis, out_lwir = self.insa(out_vis, out_lwir)
        
        # Weighted summation
        out = torch.add(out_vis * self.weight, out_lwir * (1 - self.weight))
        # print("out CHECKPOINT CHECKPOINT CHECKPOINT CHECKPOINT CHECKPOINT")
        # Final pooling
        out = self.pool3(out)
        # print("pool3 CHECKPOINT CHECKPOINT CHECKPOINT CHECKPOINT CHECKPOINT")
        print(out.shape)
        # return out
        return {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6}


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
    
    def output_shape(self):
        return {
            'p2': ShapeSpec(channels=256, stride=4),
            'p3': ShapeSpec(channels=256, stride=8),
            'p4': ShapeSpec(channels=256, stride=16),
            'p5': ShapeSpec(channels=256, stride=32),
            'p6': ShapeSpec(channels=256, stride=64),
        }


@BACKBONE_REGISTRY.register()
class DualStreamFusionBackbone(Backbone):
    def __init__(self, cfg, input_shape=None):
        super(DualStreamFusionBackbone, self).__init__()
        self.weight = 0.5
        pretrained = cfg.MODEL.BACKBONE.PRETRAINED if hasattr(cfg.MODEL.BACKBONE, 'PRETRAINED') else False

        def create_resnet34_backbone(pretrained, in_channels):
            # model = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=pretrained)
            model = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', pretrained=pretrained)
            old_conv = model.body.conv1 
            # Modify the first convolutional layer to accept 3 input channels (RGB)
            model.body.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, 
                                    kernel_size=old_conv.kernel_size, 
                                    stride=old_conv.stride, 
                                    padding=old_conv.padding, 
                                    bias=old_conv.bias)
            if pretrained and in_channels == 1:
                # Initialize the new conv layer weights for a single channel by averaging RGB weights
                weight = old_conv.weight.data.mean(dim=1, keepdim=True)
                model.body.conv1.weight.data = weight
            return model  # Exclude the avgpool and fc layers
    
        self.visible_backbone = create_resnet34_backbone(pretrained, 3)  # RGB images, 3 channels
        self.lwir_backbone = create_resnet34_backbone(pretrained, 1)  # LWIR images, 1 channel
        
        # Assuming 512 channels output from each backbone, which are then fused
        # self.fusion_layer = nn.Conv2d(512 * 2, 1024, kernel_size=1, bias=False)
        self.fusion_layer = nn.Conv2d(512, 1024, kernel_size=1, bias=False)

        # To simulate multi-level feature maps required by FPN
        self.reduce_layers = nn.ModuleDict({
            'p2': nn.Conv2d(1024, 256, kernel_size=1),
            'p3': nn.Conv2d(1024, 256, kernel_size=1),
            'p4': nn.Conv2d(1024, 256, kernel_size=1),
            'p5': nn.Conv2d(1024, 256, kernel_size=1),
            'p6': nn.Conv2d(1024, 256, kernel_size=1),
        })
        

    def forward(self, x):
        visible_img, lwir_img = x[:, 0:3, :, :], x[:, 3:4, :, :]
        # print("visible_backbone")
        visible_features = self.visible_backbone(visible_img)
        # print("lwir_backbone")
        lwir_features = self.lwir_backbone(lwir_img)
        # print(visible_features)
        # print(lwir_features)
        # print("debug")
        # print(visible_features['0'])
        # print(visible_features.shape, lwir_features.shape)
        # combined_features = torch.cat([visible_features, lwir_features], dim=1)
        # print("combined_features")

        # combined_features = torch.add(visible_features * self.weight, lwir_features * (1 - self.weight))
        # combined_features = visible_features
        # print("fused_features")
        # fused_features = self.fusion_layer(combined_features)
        # print("out")
        # Create multiple feature map levels by applying reduction layers
        out = {
            'p2': torch.add(visible_features['0'] * self.weight, lwir_features['0'] * (1 - self.weight)),
            'p3': torch.add(visible_features['1'] * self.weight, lwir_features['1'] * (1 - self.weight)),
            'p4': torch.add(visible_features['2'] * self.weight, lwir_features['2'] * (1 - self.weight)),
            'p5': torch.add(visible_features['3'] * self.weight, lwir_features['3'] * (1 - self.weight)),
            'p6': torch.add(visible_features['pool'] * self.weight, lwir_features['pool'] * (1 - self.weight)),
        }
        return out

    def output_shape(self):
        return {
            'p2': ShapeSpec(channels=256, stride=4),
            'p3': ShapeSpec(channels=256, stride=8),
            'p4': ShapeSpec(channels=256, stride=16),
            'p5': ShapeSpec(channels=256, stride=32),
            'p6': ShapeSpec(channels=256, stride=64),
        }

@BACKBONE_REGISTRY.register()
class DebugVisibleResNet50FPNBackbone(Backbone):
    def __init__(self, cfg, input_shape=None):
        super(DebugVisibleResNet50FPNBackbone, self).__init__()
        pretrained = cfg.MODEL.BACKBONE.PRETRAINED if hasattr(cfg.MODEL.BACKBONE, 'PRETRAINED') else False

        def create_resnet50_fpn_backbone(pretrained):
            model = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=pretrained)
            # Modify the first convolutional layer to accept 3 input channels (RGB)
            model.body.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            return model

        self.visible_backbone = create_resnet50_fpn_backbone(pretrained)

    def forward(self, x):
        visible_img = x[:, 0:3, :, :]  # Select the first 3 channels for the visible (RGB) input
        # print(visible_img)
        visible_features = self.visible_backbone(visible_img)
        # print(visible_features.shape)
        # print("Visible features keys:", visible_features.keys())
        # print(visible_features['0'].shape, visible_features['1'].shape, visible_features['2'].shape, visible_features['3'].shape,  visible_features['pool'].shape)
        # raise Exception("Debug")
        return {"p2": visible_features['0'], "p3": visible_features['1'], "p4": visible_features['2'], "p5": visible_features['3'], "p6": visible_features['pool']}  # Output features from multiple levels (p2, p3, p4, p5)

    def output_shape(self):
        return {
            "p2": ShapeSpec(channels=256, stride=4),
            "p3": ShapeSpec(channels=256, stride=8),
            "p4": ShapeSpec(channels=256, stride=16),
            "p5": ShapeSpec(channels=256, stride=32),
            "p6": ShapeSpec(channels=256, stride=64),
        }


@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # create your own backbone
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}