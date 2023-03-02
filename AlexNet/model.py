# -*- coding = utf-8 -*-  
# @Time: 2023/3/2 16:25 
# @Author: Dylan 
# @File: model.py 
# @software: PyCharm

import os
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    Neural Network Architecture proposed in the AlexNet paper
    """
    def __init__(self, num_classes=1000, init_weight=False):
        """
        Define and allocate layers for this neural net
        :param num_classes: num of labels predict by this model
        """
        super().__init__()
        # for conv layers
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=11,
                      stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,
                                 alpha=0.0001,
                                 beta=0.75,
                                 k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,
                                 alpha=0.0001,
                                 beta=0.75,
                                 k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # for the fully connected layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256*6*6),
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),
            nn.Linear(4096, out_features=num_classes)
        )

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        """
        forward propagation
        :param x:  input
        :return: x
        """
        x = self.net(x)  # put x into net
        x = torch.flatten(x, start_dim=1)  # x = x.view(-1, 256*6*6) reduce the dimension for the linear layer input
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 对每一层网络进行遍历
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu') # not same as paper
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)




