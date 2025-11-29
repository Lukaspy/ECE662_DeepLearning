#Implements a ResNet-18 and ResNet-50 architecture
#Lukas Crockett ECE662 Project 3

import os
import torch
from torch import nn



class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, skip_fn=None):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(out_channels)
        )
    
        self.skip_fn = skip_fn
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.residual(x)

        # pass the original input through the skip fxn
        if self.skip_fn is not None:
            identity = self.skip_fn(identity)
        
        out += identity

        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.layer1 = self

    def make_residual_layer(self, in_channels, out_channels,stride):
        #create a single residual layer from two basic blocks
        layers = []

        # If spatial size or channels change, we need a projection on the skip
        skip_fn = None
        if stride != 1 or in_channels != out_channels:
            skip_fn = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        # First block in the stage (may downsample)
        layers.append(BasicBlock(in_channels, out_channels, stride=stride, skip_fn=skip_fn))

        # Second block: same channels, stride 1, pure identity skip
        layers.append(BasicBlock(out_channels, out_channels, stride=1, skip_fn=None))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        def forward(self, x):
            x = self.stem(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)          # (N, 512, 1, 1)
            x = torch.flatten(x, 1)      # (N, 512)
            x = self.fc(x)               # (N, num_classes)
            return x

