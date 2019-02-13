'''
    File name: HANDBOOK
    Author: minhnc
    Date created(MM/DD/YYYY): 12/10/2018
    Last modified(MM/DD/YYYY HH:MM): 12/10/2018 11:18 AM
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

from data_loader import PartDataset

from model import PointNetPointWise
from torch.autograd import Variable

from points_visualization import visualize

from open3d import *
from data_loader import load_ply
from surface_matching import draw_registration_result, preprocess_point_cloud, execute_global_registration, refine_registration
from surface_matching import rotationMatrixToEulerAngles

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--np', type=int, default=4096, help='number of input points(size of input point cloud)')
parser.add_argument('--ptn', type=str, default='./tmp/seg_model_94_0.944126.pth', help='patch of pre-trained model')
parser.add_argument('--idx', type=int, default=0, help='model index')

#==============================================================================
# Function Definitions
#==============================================================================
def predict_segmentation(classifier, input_points):
    start_time_pred = time.time()

    # Prepare data to feed to PointNet
    points_ts = torch.from_numpy(input_points)
    points_ts = points_ts.transpose(1, 0).contiguous()
    points_ts = Variable(points_ts.view(1, points_ts.size()[0], points_ts.size()[1]))

    # Predict results
    pred = classifier(points_ts)
    pred_labels = pred.data.max(2)[1]  # get predicted labels
    pred_labels = pred_labels[0].data.cpu().numpy()  # convert tensor to numpy array

    stop_time_pred = time.time()
    print('Deep Network Execution Time: ', stop_time_pred - start_time_pred)
    return pred_labels


def visualize_segmented_objects(scene_points, labels):
    # Visualize segmented objects
    pipe_points = scene_points[labels==1] # Get point set of: 1-pipes, 2-wrench
    wrench_points = scene_points[labels==2] # Get point set of: 1-pipes, 2-wrench
    if len(pipe_points) > 10:
        visualize(x=pipe_points[:, 0], y=pipe_points[:, 1], z=pipe_points[:, 2], label=np.ones(len(pipe_points)), point_radius=0.0008)  # visualize segmented pipes
    if len(wrench_points) > 10:
        visualize(x=wrench_points[:, 0], y=wrench_points[:, 1], z=wrench_points[:, 2], label=np.ones(len(wrench_points)), point_radius=0.0008)  # visualize segmented pipes
    return pipe_points, wrench_points


def match_surface(source_points, target_points):
    source = PointCloud()
    source.points = Vector3dVector(source_points)
    target = PointCloud()
    target.points = Vector3dVector(target_points)
    voxel_size = 0.002

    # downsample data
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    draw_registration_result(source, target, np.identity(4))  # visualize point cloud

    # 1st: gross matching(RANSAC)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    # 2nd: fine-tune matching(ICP)
    result_icp = refine_registration(source, target, voxel_size, result_ransac)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)
    return result_icp

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is PointNet-Segmentation and Surface-Matching Program')

    opt = parser.parse_args()
    num_points = opt.np
    pretrained_model = opt.ptn
    idx = opt.idx
    root = 'E:/PROJECTS/NTUT/PointNet/pointnet1_pytorch/DATA/Shapenet/shapenetcore_partanno_segmentation_benchmark_v0'
    num_classes = 3

    '''
    Load PointNet Model (model for point-wise classification)
    '''
    classifier = PointNetPointWise(num_points=num_points, num_classes=num_classes)
    classifier.load_state_dict(torch.load(pretrained_model))
    classifier.eval()

    '''
    Load data
    '''
    # data_tes = PartDataset(root=root, num_points=num_points, categories=['tools'], training=True, balanced=True,
    #                        shuffle=True, seed=0, offset=0)
    #
    # points, labels = data_tes[idx] # Load data points and ground truth labels
    # visualize(x=points[:, 0], y=points[:, 1], z=points[:, 2], label=labels, point_radius=0.0008) # visualize ground truth

    pc_scene = load_ply(path='./models/model5/PC/pipe/1001.ply', num_points=4096)
    pc_scene = pc_scene.astype(np.float32)

    '''
    Predict and Segment objects
    '''
    pred_labels = predict_segmentation(classifier=classifier, input_points=pc_scene)
    visualize(x=pc_scene[:, 0], y=pc_scene[:, 1], z=pc_scene[:, 2], label=pred_labels, point_radius=0.0008)  # visualize predicted results

    pipe_points, wrench_points = visualize_segmented_objects(scene_points=pc_scene, labels=pred_labels)

    '''
    Surface-Matching
    '''
    # Convert numpy array to point cloud type
    pc_model = load_ply(path='./models/model5/PC/pipe/1002.ply', num_points=-1)

    # pc_scene = wrench_points
    visualize(x=pc_model[:, 0], y=pc_model[:, 1], z=pc_model[:, 2], label=np.ones(len(pc_model)), point_radius=0.0008)
    visualize(x=pipe_points[:, 0], y=pipe_points[:, 1], z=pipe_points[:, 2], label=np.ones(len(pipe_points)), point_radius=0.0008)

    # result_icp = match_surface(source_points=pc_model, target_points=pipe_points); print(result_icp.transformation[:3, 3])
    result_icp = match_surface(source_points=pc_model, target_points=pc_scene); print(result_icp.transformation[:3, 3])

    # Transformation(Rotation angles)
    print('Theta x, Theta y, Theta z:(in Degree) ')
    print(rotationMatrixToEulerAngles(result_icp.transformation[:3, :3]) / np.pi * 180)

if __name__ == '__main__':
    main()
