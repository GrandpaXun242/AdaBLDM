from futils import FeaturesCollector, load_ckpt
from timm.models import densenet
from torchvision.models import efficientnet
from torchvision.models import mobilenet
from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List, Dict, Union, Tuple
import torch
import torch.nn as nn
  
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=.001)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 64, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class BaseModel(nn.Module):
    def __init__(self, output_dim: int = 384, backbone_name: str = 'densenet201', pretrained: bool = True, freeze_pre_bn: bool = False, layers: List[str] = ['features.denseblock1', 'features.denseblock2']):
        """
        Args:
            backbone_name (str, optional): backbone. Defaults to 'densenet201'.
            pretrained (bool, optional): 是否加载预训练模型. Defaults to True.
            freeze_pre_bn (bool, optional): 是否冻结之前的所有bn层. Defaults to False.
            layers (List[str], optional): 使用的特征层
        """  
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.freeze_pre_bn = freeze_pre_bn
        self.layers = layers
        self.freeze_modules = []
        self.backbone = self.load_backbone(backbone_name, pretrained)
        # self.features_collector = create_feature_extractor(self.load_backbone(backbone_name, pretrained), {layer: layer for layer in self.layers})
        self.freeze()
        self.features_collector = FeaturesCollector(self.backbone, layers, True)
        self.shapes_d_ = self.get_features(torch.randn(4, 3, 320, 320))
        self.conv = nn.ModuleDict({
            f'conv_{i}': nn.Sequential(
                nn.Conv2d(self.shapes_d_[layer].shape[1], 192, kernel_size=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                Mixed_5b(),
                nn.Conv2d(4 * 64, output_dim, kernel_size=1)
            ) for i, layer in enumerate(layers)
        })
    
    def freeze(self):
        """冻结当前为止之前的所有网络层
        """        
        self.requires_grad_(False)
        self.freeze_modules = [m for m in self.children()]
        
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_pre_bn:  # 固定batchnorm和dropout层
            for m in self.freeze_modules:
                m.eval()
    
    def forward(self, x) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        p_features_d = self.get_features(x)
        ret = []
        for i, (layer, features) in enumerate(p_features_d.items()):
            ret.append(self.conv[f'conv_{i}'](features))
        if self.training:
            return ret
        else:
            return ret, p_features_d

    @torch.no_grad()
    def get_features(self, x) -> Dict[str, torch.Tensor]:
        return self.features_collector(x)
    
    def load_backbone(self, backbone_name, pretrained):
        raise NotImplementedError("load backbone must be implemented")


class DensenetPM(BaseModel):
    def __init__(self, output_dim: int = 384, backbone_name: str = 'densenet201', pretrained: bool = True, freeze_pre_bn: bool = False, layers: List[str] = ['features.denseblock1', 'features.denseblock2']):
        super().__init__(output_dim, backbone_name, pretrained, freeze_pre_bn, layers)
    
    def load_backbone(self, backbone_name, pretrained) -> densenet.DenseNet:
        return getattr(densenet, backbone_name)(pretrained=pretrained)


class ResnetPM(BaseModel):
    def __init__(self, output_dim: int = 384, backbone_name: str = 'resnet18', pretrained: bool = True, freeze_pre_bn: bool = False, layers: List[str] = ['layer1', 'layer2']):
        super().__init__(output_dim, backbone_name, pretrained, freeze_pre_bn, layers)
    
    def load_backbone(self, backbone_name, pretrained) -> resnet.ResNet:
        return getattr(resnet, backbone_name)(pretrained=pretrained)


class EfficientnetPM(BaseModel):
    def __init__(self, output_dim: int = 384, backbone_name: str = 'efficientnet_b0', pretrained: bool = True, freeze_pre_bn: bool = False, layers: List[str] = ['features.2', 'features.3']):
        super().__init__(output_dim, backbone_name, pretrained, freeze_pre_bn, layers)
    
    def load_backbone(self, backbone_name, pretrained) -> efficientnet.EfficientNet:
        return getattr(efficientnet, backbone_name)(pretrained=pretrained)


class MobilenetPM(BaseModel):
    def __init__(self, output_dim: int = 384, backbone_name: str = 'mobilenet_v3_small', pretrained: bool = True, freeze_pre_bn: bool = False, layers: List[str] = ['features.2.block.0', 'features.4.block.0']):
        super().__init__(output_dim, backbone_name, pretrained, freeze_pre_bn, layers)
    
    def load_backbone(self, backbone_name, pretrained) -> mobilenet.MobileNetV3:
        return getattr(mobilenet, backbone_name)(pretrained=pretrained)

