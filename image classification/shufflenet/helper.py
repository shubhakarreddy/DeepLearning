import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def conv1(input_channels, output_channels, groups, stride):
    return nn.Conv2d(
        in_channels=input_channels, out_channels=output_channels,
        kernel_size=3, stride=stride, padding=1,
        groups=groups, bias=True, padding_mode='zeros'
    )

class ShuffleNetUnit(nn.Module):
    def __init__(self, input_size, output_size, stride, groups, use_groups=True):
        super(ShuffleNetUnit, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bottleneck_size = int(output_size/4)
        self.stride = stride
        self.groups = groups if use_groups else 1

        # Adjust output size in case of shortcut path concatenation
        if self.stride > 1:
            self.output_size -= self.input_size

        self.gconv1x1_before = nn.Sequential(
                                nn.Conv2d(in_channels=self.input_size, out_channels=self.bottleneck_size,
                                          groups=self.groups, kernel_size=1, stride=1),
                                nn.BatchNorm2d(num_features=self.bottleneck_size),
                                nn.ReLU()
        )

        self.dwconv3x3 = nn.Sequential(
                            nn.Conv2d(in_channels=self.bottleneck_size, out_channels=self.bottleneck_size,
                                      groups=self.bottleneck_size, kernel_size=3, stride=self.stride, padding=1),
                            nn.BatchNorm2d(num_features=self.bottleneck_size)
        )

        self.gconv1x1_after = nn.Sequential(
                                nn.Conv2d(in_channels=self.bottleneck_size, out_channels=self.output_size,
                                          groups=self.groups, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.output_size)
        )

        self.shortcut_path = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def shuffle(self, input):
        N, C, H, W = input.size()
        # Break down C into groups * C' and then transpose
        input = input.view(N, self.groups, C//self.groups, H, W)
        input = torch.transpose(input, 1, 2).contiguous()
        input = input.view(N, C, H, W)

        return input

    def forward(self, input):
        out = self.gconv1x1_before(input)
        out = self.shuffle(out)
        out = self.dwconv3x3(out)
        out = self.gconv1x1_after(out)

        if self.stride > 1:
            return F.relu(torch.cat((self.shortcut_path(input), out), 1))
        else:
            return F.relu(input + out)
