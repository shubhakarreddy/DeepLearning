import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import helper

class ResNet(nn.Module):
    # in_channels = 3, out_channels = {stage: out_channels_count}
    def __init__(self, in_channels, out_channels, repeats, num_class):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class = num_class

        self.stage_1 = nn.Sequential(
                helper.conv1(self.in_channels, self.out_channels[1], 2),
                nn.BatchNorm2d(num_features=self.out_channels[1]),
                nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_2_eles = [helper.ResNetUnit(out_channels[1], out_channels[2], 1)]
        for i in range(repeats[2]-1):
            stage_2_eles.append(helper.ResNetUnit(out_channels[2], out_channels[2],
                                1))
        self.stage_2 = nn.Sequential(*stage_2_eles)

        stage_3_eles = [helper.ResNetUnit(out_channels[2], out_channels[3], 2)]
        for i in range(repeats[3]-1):
            stage_3_eles.append(helper.ResNetUnit(out_channels[3], out_channels[3],
                                1))
        self.stage_3 = nn.Sequential(*stage_3_eles)

        stage_4_eles = [helper.ResNetUnit(out_channels[3], out_channels[4], 2)]
        for i in range(repeats[4]-1):
            stage_4_eles.append(helper.ResNetUnit(out_channels[4], out_channels[4],
                                1))
        self.stage_4 = nn.Sequential(*stage_4_eles)

        stage_5_eles = [helper.ResNetUnit(out_channels[4], out_channels[5], 2)]
        for i in range(repeats[5]-1):
            stage_5_eles.append(helper.ResNetUnit(out_channels[5], out_channels[5],
                                1))
        self.stage_5 = nn.Sequential(*stage_5_eles)

        self.fc = nn.Linear(out_channels[5], num_class, bias=False)

        # Loss
        self.linear_closs = nn.Linear(out_channels[5], 1000, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, input):
        # Stage 1
        out = self.maxpool(self.stage_1(input))
        # Stage 2
        out = self.stage_2(out)
        # Stage 3
        out = self.stage_3(out)
        # Stage 4
        out = self.stage_4(out)
        # Stage 5
        out = self.stage_5(out)

        out = out.reshape(out.size()[0], -1)
        label_output = self.fc(out)
        label_output = label_output/torch.norm(self.fc.weight, dim=1)

        # Create the feature embedding for the Center Loss
        # closs_output = self.linear_closs(out)
        # closs_output = self.relu_closs(closs_output)

        return [], label_output
