import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def conv1(input_channels, output_channels, stride):
    return nn.Conv2d(
        in_channels=input_channels, out_channels=output_channels,
        kernel_size=7, stride=stride, padding=3, bias=False, padding_mode='zeros'
    )

class ResNetUnit(nn.Module):
    def __init__(self, input_size, output_size, stride):
        super(ResNetUnit, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.intermediate_size = self.output_size//4
        self.stride = stride

        self.gconv1x1_before = nn.Sequential(
                                nn.Conv2d(in_channels=self.input_size, out_channels=self.intermediate_size,
                                          kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(num_features=self.intermediate_size),
                                nn.ReLU(inplace=True)
        )

        self.dwconv3x3 = nn.Sequential(
                            nn.Conv2d(in_channels=self.intermediate_size, out_channels=self.intermediate_size,
                                      kernel_size=3, stride=self.stride, padding=1, bias=False),
                            nn.BatchNorm2d(num_features=self.intermediate_size),
                            nn.ReLU(inplace=True)
        )

        self.gconv1x1_after = nn.Sequential(
                                nn.Conv2d(in_channels=self.intermediate_size, out_channels=self.output_size,
                                          kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(self.output_size)
        )

        self.downsample = nn.Sequential(
                            nn.Conv2d(in_channels=self.input_size, out_channels=self.output_size,
                                      kernel_size=1, stride=self.stride, bias=False),
                            nn.BatchNorm2d(self.output_size)
        )

    def forward(self, input):
        out = self.gconv1x1_before(input)
        out = self.dwconv3x3(out)
        out = self.gconv1x1_after(out)

        if input.size() != out.size():
            input = self.downsample(input)

        return F.relu(out + input)
