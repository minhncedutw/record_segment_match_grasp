'''
    File name: HANDBOOK
    Author: minhnc
    Date created(MM/DD/YYYY): 12/4/2018
    Last modified(MM/DD/YYYY HH:MM): 12/4/2018 7:40 AM
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import math
import numpy as np
import torch

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
import torch.utils.data as data
class PartDataset(data.Dataset):
    def __init__(self, root, num_points=2048, categories=None, training=True, balanced=False, shuffle=False, seed=0, offset=0):
        self.root = root
        self.num_points = num_points
        self.categories = categories
        self.offset = offset
        self.balanced = balanced
        self.shuffle = shuffle
        self.seed = seed
        self.category_file = os.path.join(self.root, 'synsetoffset2category.txt')
        self.categories = {}

        with open(self.category_file, 'r') as f:
            for line in f:
                category, folder = line.strip().split()
                self.categories[category] = folder

        if not categories is  None:
            self.categories = {k:v for k,v in self.categories.items() if k in categories}

        self.meta = {}
        for category in self.categories:
            self.meta[category] = []
            dir_point = os.path.join(self.root, self.categories[category], 'points')
            dir_label = os.path.join(self.root, self.categories[category], 'points_label')

            filenames = sorted(os.listdir(dir_point))
            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(filenames)
            if training:
                filenames = filenames[:int(len(filenames) * 0.8)]
            else:
                filenames = filenames[int(len(filenames) * 0.8):]

            for filename in filenames:
                token = (os.path.splitext(os.path.basename(filename))[0])
                self.meta[category].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_label, token + '.seg')))

        self.data_paths = []
        for category in self.categories:
            for file_point, file_label in self.meta[category]:
                self.data_paths.append((category, file_point, file_label))

        self.classes = dict(zip(sorted(self.categories), range(len(self.categories))))
        print(self.classes)
        self.num_seg_classes = 0
        for i in range(math.ceil(len(self.data_paths) / 50)):
            l = len(np.unique(np.loadtxt(self.data_paths[i][-1]).astype(np.uint8)))
            if l > self.num_seg_classes:
                self.num_seg_classes = l
        pass

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        category, file_point, file_label = self.data_paths[index]
        points = np.loadtxt(file_point).astype(np.float32)
        labels = np.loadtxt(file_label).astype(np.int64)

        if self.balanced:
            obj_idxs = np.argwhere(labels > 1).ravel()
            grd_idxs = np.argwhere(labels == 1).ravel()
            choice = np.array([])
            for i in range(1000):
                taken_grd_idx = np.random.choice(a=grd_idxs, size=len(obj_idxs), replace=True)
                choice = np.hstack((choice, obj_idxs))
                choice = np.hstack((choice, taken_grd_idx))
                if len(choice) >= self.num_points:
                    break
            np.random.shuffle(choice)
            choice = choice[:self.num_points].astype(np.int)
        else:
            choice = np.random.choice(len(labels), self.num_points, replace=True)

        points = points[choice, :]
        labels = labels[choice] + self.offset

        return points, labels

from open3d import *
def load_ply(path, num_points=-1):
    pointcloud = read_point_cloud(path)
    scene_array = np.asarray(pointcloud.points)

    if num_points > 0:
        choice = np.random.choice(a=len(scene_array), size=num_points, replace=True)
        scene_array = scene_array[choice, :]

    return scene_array

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
