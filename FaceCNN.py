import pandas as pd
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import utils

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            # input(bitch_size, 1, 48, 48)
            # output(bitch, 64, 24, 24)
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            
            nn.BatchNorm2d(num_features = 64),
            nn.RReLU(inplace = True),

            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.conv2 = nn.Sequential(
            # input 64, 24, 24 output 128, 12, 12
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),

            nn.BatchNorm2d(num_features = 128),
            nn.RReLU(inplace = True),

            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.conv3 = nn.Sequential(
            # input 128, 12, 12 output 256, 6, 6
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),

            nn.BatchNorm2d(num_features = 256),
            nn.RReLU(inplace = True),

            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(in_features = 256*6*6, out_features = 4096),
            nn.RReLU(inplace = True),

            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 4096, out_features = 1024),
            nn.RReLU(inplace = True),

            nn.Linear(in_features = 1024, out_features = 256),
            nn.RReLU(inplace = True),

            nn.Linear(in_features = 256, out_features = 7),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y