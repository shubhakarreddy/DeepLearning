import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import helper

class ShuffleNet(nn.Module):
    # in_channels = 3, out_channels = {stage: out_channels_count}
    def __init__(self, in_channels, out_channels, groups, repeats, num_class):
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class = num_class

        self.dropout = nn.Dropout2d(p=0.5)
        self.conv1 = helper.conv1(self.in_channels, self.out_channels[1], 1, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_2_eles = [helper.ShuffleNetUnit(out_channels[1], out_channels[2], 2, self.groups, use_groups=False)]
        for i in range(repeats[2]):
            stage_2_eles.append(helper.ShuffleNetUnit(out_channels[2], out_channels[2],
                                1, self.groups))
        self.stage_2 = nn.Sequential(*stage_2_eles)

        stage_3_eles = [helper.ShuffleNetUnit(out_channels[2], out_channels[3], 2, self.groups, use_groups=False)]
        for i in range(repeats[3]):
            stage_3_eles.append(helper.ShuffleNetUnit(out_channels[3], out_channels[3],
                                1, self.groups))
        self.stage_3 = nn.Sequential(*stage_3_eles)

        stage_4_eles = [helper.ShuffleNetUnit(out_channels[3], out_channels[4], 2, self.groups, use_groups=False)]
        for i in range(repeats[4]):
            stage_4_eles.append(helper.ShuffleNetUnit(out_channels[4], out_channels[4],
                                1, self.groups))
        self.stage_4 = nn.Sequential(*stage_4_eles)

        # mlp_hidden_units = [960, 1024, 1024, 2048, 2048, 2300]
        # mlp_eles = [nn.Linear(mlp_hidden_units[0], mlp_hidden_units[1])]
        # for i in range(1, len(mlp_hidden_units)-1):
        #     mlp_eles.append(nn.BatchNorm1d(num_features=mlp_hidden_units[i]))
        #     mlp_eles.append(nn.ReLU())
        #     mlp_eles.append(nn.Linear(mlp_hidden_units[i], mlp_hidden_units[i+1]))

        # self.mlp = nn.Sequential(*mlp_eles)

        self.fc = nn.Linear(out_channels[4], num_class, bias=False)

        # Loss
        self.linear_closs = nn.Linear(out_channels[4], 1000, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, input):
        # Stage 1
        out = self.maxpool(self.conv1(self.dropout(input)))
        # Stage 2
        out = self.stage_2(out)
        # Stage 3
        out = self.stage_3(out)
        # Stage 4
        out = self.stage_4(out)

        out = out.reshape(out.size()[0], -1)
        label_output = self.fc(out)
        # label_output = label_output/torch.norm(self.mlp[-1].weight, dim=1)

        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(out)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output
