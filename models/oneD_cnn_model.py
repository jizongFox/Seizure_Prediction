# -*- coding: utf-8 -*-
from torch import nn

class oneD_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=7,dilation=2,stride=7),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=7,dilation=2,stride=7),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=7,dilation=2,stride=7),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=640, out_features=2048),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,out_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,out_features=2)
        )
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output=output.view(output.size(0),-1)
        output = self.linear1(output)
        return output


