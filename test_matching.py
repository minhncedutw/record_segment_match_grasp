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
    Orbbec camera calibration tool at: https://3dclub.orbbec3d.com/t/astra-s-intrinsic-and-extrinsic-parameters/302/4
                                   or: https://3dclub.orbbec3d.com/t/universal-download-thread-for-astra-series-cameras/622
    Orbbec Astra S FoV: 60° horiz x 49.5° vert. (73° diagonal) (https://orbbec3d.com/product-astra/)
    Camera intrinsic: http://ksimek.github.io/2013/08/13/intrinsic/
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

from primesense import openni2
from primesense import _openni2 as c_api

import cv2

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--np', type=int, default=8192, help='number of input points(size of input point cloud)')
parser.add_argument('--ptn', type=str, default='./tmp/seg_model_94_0.944126.pth', help='patch of pre-trained model')
parser.add_argument('--idx', type=int, default=0, help='model index')

#==============================================================================
# Function Definitions
#==============================================================================
def setup_camera(w, h, fps):
    ## Initialize OpenNi
    # dist = './driver/OpenNI-Linux-x64-2.3/Redist'
    dist = './driver/OpenNI-Windows-x64-2.3/Redist'
    openni2.initialize(dist)
    if (openni2.is_initialized()):
        print("openNI2 initialized")
    else:
        print("openNI2 not initialized")

    ## Register the device
    dev = openni2.Device.open_any()

    ## Create the streams stream
    rgb_stream = dev.create_color_stream()
    depth_stream = dev.create_depth_stream()

    ## Configure the rgb_stream -- changes automatically based on bus speed
    rgb_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=w, resolutionY=h,
                           fps=fps))

    ## Configure the depth_stream -- changes automatically based on bus speed
    # print 'Depth video mode info', depth_stream.get_video_mode() # Checks depth video configuration
    depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=w, resolutionY=h,
                           fps=fps))

    ## Check and configure the mirroring -- default is True
    ## Note: I disable mirroring
    # print 'Mirroring info1', depth_stream.get_mirroring_enabled()
    depth_stream.set_mirroring_enabled(False)
    rgb_stream.set_mirroring_enabled(False)

    ## Start the streams
    rgb_stream.start()
    depth_stream.start()

    ## Synchronize the streams
    dev.set_depth_color_sync_enabled(True)  # synchronize the streams

    ## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    return rgb_stream, depth_stream


def get_rgb(rgb_stream, h, w):
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(h, w, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_depth(depth_stream, h, w):
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
    Note1:
        fromstring is faster than asarray or frombuffer
    Note2:
        .reshape(120,160) #smaller image for faster response
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(h, w)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
    return dmap, d4d


def display_stream(rgb_stream, depth_stream, h, w, crop=((200, 140), (440, 340)), h1=140, h2=340, w1=200, w2=440):
    if crop is not None:
        h1 = crop[0][1]
        h2 = crop[1][1]
        w1 = crop[0][0]
        w2 = crop[1][0]

    ## Stream
    # RGB
    rgb = get_rgb(rgb_stream=rgb_stream, h=h, w=w)

    # DEPTH
    dmap, d4d = get_depth(depth_stream=depth_stream, h=h, w=w)

    # canvas
    canvas = np.hstack((rgb, d4d))
    cv2.rectangle(canvas, (w1, h1), (w2, h2), (0, 255, 0), 1)
    cv2.rectangle(canvas, (w1 + w, h1), (w2 + w, h2), (0, 255, 0), 1)
    cv2.line(img=canvas, pt1=(200, 240), pt2=(440, 240), color=(255, 0, 0), thickness=1)
    cv2.line(img=canvas, pt1=(320, 140), pt2=(320, 340), color=(255, 0, 0), thickness=1)

    ## Display the stream syde-by-side
    cv2.imshow('depth || rgb', canvas)
    return rgb, dmap

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


def match_surface(model_points, object_points):
    source = PointCloud()
    source.points = Vector3dVector(model_points)
    target = PointCloud()
    target.points = Vector3dVector(object_points)
    voxel_size = 0.001

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

# def rgbd2pc(rgb, depth, center_x=0, center_y=0, focal_length=890, scale=1000):
#     points = []
#     for v in range(rgb.shape[1]):
#         for u in range(rgb.shape[0]):
#             color = rgb[u, v]
#             Z = depth[u, v] / scale
#             if Z == 0:
#                 continue
#             X = (u - center_x) * Z / focal_length
#             Y = (v - center_y) * Z / focal_length
#
#             points.append([X, Y, Z, color[0], color[1], color[2], 0])
#     return points

def rgbd2xyzrgbrc(rgb, depth, center_x=None, center_y=None, focal_length=889, scale=1000, phiX=0.1061, phiY=0.1082):
    H, W = depth.shape
    if center_x==None: center_x = (W-1)/2
    if center_y==None: center_y = (H-1)/2
    xyzrgbrc = np.zeros(shape=(H, W, 9))
    for c in range(rgb.shape[1]):
        for r in range(rgb.shape[0]):

            Z = depth[r, c]
            if Z==0: mask = 0
            else: mask = 1

            thetaX = phiX * (c - center_x) * np.pi / 180
            thetaY = phiY * (r - center_y) * np.pi / 180

            X = np.tan(thetaX) * Z
            Y = np.tan(thetaY) * Z

            X = X / scale
            Y = Y / scale
            Z = Z / scale

            R, G, B = rgb[r, c]

            xyzrgbrc[r, c] = [X, Y, Z, R, G, B, mask, r, c]
    return xyzrgbrc

def xyzrgb2pc(xyzrgb):
    H, W, _ = xyzrgb.shape
    points = []
    for v in range(W):
        for u in range(H):
            X, Y, Z, R, G, B = xyzrgb[u, v, 0:6]
            if int(xyzrgb[u, v, 6])==0: continue # ignore point that mask=0 (it means that Z=0 and it is unrecorded-able point
            points.append([X, Y, Z, R, G, B, 0])
    return points


def generate_ply(points):
    points_str = []
    for i in range(len(points)):
        points_str.append(f"{points[i][0]:f} {points[i][1]:f} {points[i][2]:f} {points[i][3].astype(np.int):d} {points[i][4].astype(np.int):d} {points[i][5].astype(np.int):d} 0\n")
    ply = f"""\
ply
format ascii 1.0
element vertex {len(points_str):d}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
{''.join(points_str)}\
"""
    return ply

def sample_data(point_cloud, num_points=-1):
    point_cloud = np.array(point_cloud)[:, :3].astype(np.float32)

    if num_points > 0:
        choice = np.random.choice(a=len(point_cloud), size=num_points, replace=True)
        point_cloud = point_cloud[choice, :]
    return point_cloud

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is PointNet-Segmentation and Surface-Matching Program')

    opt = parser.parse_args()
    num_points = opt.np
    pretrained_model = opt.ptn
    num_classes = 3

    '''
    Load PointNet Model (model for point-wise classification)
    '''
    classifier = PointNetPointWise(num_points=num_points, num_classes=num_classes)
    classifier.load_state_dict(torch.load(pretrained_model))
    classifier.eval()

    '''
    Setup camera
    '''
    fps = 30
    w, h, h1, h2, w1, w2 = (np.array([640, 480, 150, 330, 210, 430])).astype(int)

    rgb_stream, depth_stream = setup_camera(w=w, h=h, fps=fps)

    # Load model
    pc_model = load_ply(path='./models/model13_1.ply', num_points=-1)

    '''
    Record
    '''
    done = False
    while not done:
        key = cv2.waitKey(1) & 255
        ## Read keystrokes
        if key == 27:  # terminate
            print("\tESC key detected!")
            done = True
        elif chr(key) == 's':  # screen capture
            '''
            Get data
            '''
            # sTime = time.time()
            # xyzrgbrc1 = rgbd2xyzrgbrc(rgb=rgb, depth=dmap, scale=1000)
            # xyzrgbrc1 = xyzrgbrc1[h1:h2, w1:w2, :] # crop the interested area
            # print(time.time() - sTime)

            sTime = time.time()
            xyzrgbrc2 = rgbd2xyzrgbrc(rgb=rgb[h1:h2, w1:w2, :], depth=dmap[h1:h2, w1:w2], scale=1000)
            print(time.time() - sTime)

            pc_scene = xyzrgb2pc(xyzrgb=xyzrgbrc)
            pc_scene = sample_data(point_cloud=pc_scene, num_points=num_points)

            '''
            Predict and Segment objects
            '''
            pred_labels = predict_segmentation(classifier=classifier, input_points=pc_scene)
            visualize(x=pc_scene[:, 0], y=pc_scene[:, 1], z=pc_scene[:, 2], label=pred_labels,
                      point_radius=0.0008)  # visualize predicted results
            pipe_points, wrench_points = visualize_segmented_objects(scene_points=pc_scene,
                                                                     labels=pred_labels)  # visualize segmented objects

            '''
            Surface-Matching
            '''
            # pc_scene = wrench_points
            visualize(x=pc_model[:, 0], y=pc_model[:, 1], z=pc_model[:, 2], label=np.ones(len(pc_model)), point_radius=0.0008)
            visualize(x=pipe_points[:, 0], y=pipe_points[:, 1], z=pipe_points[:, 2], label=np.ones(len(pipe_points)), point_radius=0.0008)

            result_icp = match_surface(model_points=pc_model, object_points=pipe_points)

            # Transformation(Rotation angles)
            print('Theta x, Theta y, Theta z:(in Degree) ')
            print(rotationMatrixToEulerAngles(result_icp.transformation[:3, :3]) / np.pi * 180)

        rgb, dmap = display_stream(rgb_stream=rgb_stream, depth_stream=depth_stream, h=h, w=w, crop=((w1, h1), (w2, h2)))
    # end while

    ## Release resources
    cv2.destroyAllWindows()
    rgb_stream.stop()
    depth_stream.stop()
    openni2.unload()
    print("Terminated")

if __name__ == '__main__':
    main()
