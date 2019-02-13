'''
    File name: HANDBOOK
    Author: minhnc
    Date created(MM/DD/YYYY): 12/4/2018
    Last modified(MM/DD/YYYY HH:MM): 12/4/2018 7:50 AM
    Python Version: 3.6
    Other modules: [None]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import torch

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class InputTransformation(nn.Module):
    def __init__(self, num_points):
        super(InputTransformation, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.mp1 = torch.nn.MaxPool1d(kernel_size=num_points)
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=9)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        pass

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class FeatureTransformation(nn.Module):
    def __init__(self, num_points):
        super(FeatureTransformation, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.mp1 = torch.nn.MaxPool1d(kernel_size=num_points)
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=64*64)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        pass

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(64).flatten().astype(np.float32))).view(1, 64*64).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 64, 64)
        return x

class PointNetFeature(nn.Module):
    def __init__(self, num_points, global_feature=True):
        super(PointNetFeature, self).__init__()
        self.transform = InputTransformation(num_points=num_points)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.mp1 = nn.MaxPool1d(kernel_size=num_points)

        self.num_point = num_points
        self.global_feature = global_feature
        pass

    def forward(self, x):
        transformation = self.transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        point_feature = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feature:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_point)
            return torch.cat([x, point_feature], 1)

class PointNetFullFeature(nn.Module):
    def __init__(self, num_points, global_feature=True):
        super(PointNetFullFeature, self).__init__()
        self.input_transform = InputTransformation(num_points=num_points)
        self.conv11 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv12 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)

        self.feature_transform = FeatureTransformation(num_points=num_points)
        self.conv21 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv22 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv23 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.bn11 = nn.BatchNorm1d(num_features=64)
        self.bn12 = nn.BatchNorm1d(num_features=64)

        self.bn21 = nn.BatchNorm1d(num_features=64)
        self.bn22 = nn.BatchNorm1d(num_features=128)
        self.bn23 = nn.BatchNorm1d(num_features=1024)

        self.mp1 = nn.MaxPool1d(kernel_size=num_points)

        self.num_point = num_points
        self.global_feature = global_feature
        pass

    def forward(self, x):
        transformation = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        x = x.transpose(2, 1)
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))

        transformation = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transformation)
        x = x.transpose(2, 1)
        point_feature = x
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = self.bn23(self.conv23(x))
        x = self.mp1(x)

        x = x.view(-1, 1024)
        if self.global_feature:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_point)
            return torch.cat([x, point_feature], 1)

class PointNetPointWise(nn.Module):
    def __init__(self, num_points, num_classes):
        super(PointNetPointWise, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        # self.feature = PointNetFeature(num_points=num_points, global_feature=False)
        self.feature = PointNetFullFeature(num_points=num_points, global_feature=False)
        self.conv1 = nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=self.num_classes, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        pass

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(input=x.view(-1, self.num_classes), dim=-1)
        x = x.view(batch_size, self.num_points, self.num_classes)
        return x

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    if argv is None:
        argv = sys.argv

    if len(argv) > 1:
        for i in range(len(argv) - 1):
            print(argv[i + 1])


if __name__ == '__main__':
    main()
