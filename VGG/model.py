# -*- coding = utf-8 -*-  
# @Time: 2023/3/21 04:54 
# @Author: Dylan 
# @File: model.py 
# @software: PyCharm
from typing import Any, cast, Dict, List, Optional, Union
import torch.nn as nn
import torch

"""
Try to build the VGG model from two parts. 
- Convolutional Layers 
- Fully Connected Layers and softmax
"""
class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_class: int = 1000, init_weights: bool=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_class)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)


cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, 'M', 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, 'M', 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


def make_features(cfg: List[Union[str, int]], batch_norm=False) -> nn.Sequential:
    layers = []
    in_channels = 3

    for v in cfg:
        if v =="M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg(model_name = 'vgg16', **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model {} not in cfgs dict".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)

    return model





