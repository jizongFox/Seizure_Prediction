# -*- coding: utf-8 -*-
import torch as t, torch.nn as nn
import numpy as np, pandas as pd, matplotlib.pyplot as plt


class convNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(7,1),stride=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,6),stride=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128
                      ,out_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128,2)
        )

        for ii in self.classifier.children():
            if isinstance(ii, nn.Linear):
                t.nn.init.xavier_uniform_(ii.weight)
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(output.size(0),-1)
        output = self.classifier(output)
        return output





if __name__=='__main__':
    images = t.randn([32, 16, 7, 10])
    net = convNet()
    print(net(images))


