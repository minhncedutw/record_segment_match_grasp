'''
    File name: PointNet Definition
    Author: minhnc
    Date created(MM/DD/YYYY): 10/2/2018
    Last modified(MM/DD/YYYY HH:MM): 10/2/2018 5:25 AM
    Python Version: 3.6
    Other modules: [None]
    Reference for loss: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d#file-keras_weighted_categorical_crossentropy-py

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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1"

import numpy as np

import tensorflow as tf

import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape, LSTM
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Activation, Flatten, Dropout
from keras.layers import Lambda, concatenate
from keras.regularizers import l2

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
def exp_dim0(global_feature, axis):
    return tf.expand_dims(global_feature, axis)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

def PointNet(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    input_points = Input(shape=(num_points, 3))

    x = Convolution1D(64, 1, activation='relu',
                      input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=num_points)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    ## forward net
    g = keras.layers.dot(inputs=[input_points, input_T], axes=2)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    ## feature transformation net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    # f = MaxPooling1D(pool_size=num_points)(f)
    f = GlobalMaxPooling1D()(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    ## forward net
    g = keras.layers.dot(inputs=[g, feature_T], axes=2)
    seg_part1 = g
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    ## global_feature
    # global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = GlobalMaxPooling1D()(g)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([seg_part1, global_feature])
    c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Convolution1D(num_classes, 1, activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=input_points, outputs=prediction)
    print(model.summary())

    return model

def PointNetFull(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    input_points = Input(shape=(num_points, 3))

    x = Convolution1D(64, 1, padding='same', activation='relu',
                      input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=num_points)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    ## forward net
    g = keras.layers.dot(inputs=[input_points, input_T], axes=2)
    g64 = Convolution1D(64, 1, padding='same', input_shape=(num_points, 3), activation='relu')(g)
    g64 = BatchNormalization()(g64)
    g1281 = Convolution1D(128, 1, padding='same', input_shape=(num_points, 3), activation='relu')(g64)
    g1281 = BatchNormalization()(g1281)
    g1282 = Convolution1D(128, 1, padding='same', input_shape=(num_points, 3), activation='relu')(g1281)
    g1282 = BatchNormalization()(g1282)

    ## feature transformation net
    f = Convolution1D(128, 1, padding='same', activation='relu')(g1282)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, padding='same', activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, padding='same', activation='relu')(f)
    f = BatchNormalization()(f)
    # f = MaxPooling1D(pool_size=num_points)(f)
    f = GlobalMaxPooling1D()(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(128 * 128, weights=[np.zeros([256, 128 * 128]), np.eye(128).flatten().astype(np.float32)])(f)
    feature_T = Reshape((128, 128))(f)

    ## forward net
    g = keras.layers.dot(inputs=[g1282, feature_T], axes=2)
    seg_part1 = g
    # g = Convolution1D(64, 1, padding='same', activation='relu')(g)
    # g = BatchNormalization()(g)
    g512 = Convolution1D(512, 1, padding='same', activation='relu')(g)
    g512 = BatchNormalization()(g512)
    g = Convolution1D(2048, 1, padding='same', activation='relu')(g512)
    g = BatchNormalization()(g)

    ## global_feature
    # global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = GlobalMaxPooling1D()(g)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([g64, g1281, g1282, seg_part1, g512, global_feature])
    c = Convolution1D(512, 1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Convolution1D(num_classes, 1, padding='same', activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=input_points, outputs=prediction)
    print(model.summary())

    return model

def PointNet2(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    input_points = Input(shape=(num_points, 3))

    x = Convolution1D(64, 3, padding='same', activation='relu',
                      input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=num_points)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    ## forward net
    g = keras.layers.dot(inputs=[input_points, input_T], axes=2)
    g = Convolution1D(64, 3, padding='same', input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 3, padding='same', input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    ## feature transformation net
    f = Convolution1D(64, 3, padding='same', activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 3, padding='same', activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 3, padding='same', activation='relu')(f)
    f = BatchNormalization()(f)
    # f = MaxPooling1D(pool_size=num_points)(f)
    f = GlobalMaxPooling1D()(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    ## forward net
    g = keras.layers.dot(inputs=[g, feature_T], axes=2)
    seg_part1 = g
    g = Convolution1D(64, 3, padding='same', activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 3, padding='same', activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 3, padding='same', activation='relu')(g)
    g = BatchNormalization()(g)

    ## global_feature
    # global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = GlobalMaxPooling1D()(g)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([seg_part1, global_feature])
    c = Convolution1D(512, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=input_points, outputs=prediction)
    print(model.summary())

    return model

def mPointNet111(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    inputs = Input(shape=(num_points, 3))

    kernel_size = 3

    x = Convolution1D(64, kernel_size=kernel_size, strides=1, padding='same', input_shape=(num_points, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(64, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x)

    x = Convolution1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Convolution1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
    x2 = BatchNormalization()(x)

    x = Convolution1D(256, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x2)
    x = BatchNormalization()(x)
    x = Convolution1D(256, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
    x3 = BatchNormalization()(x)

    x = Convolution1D(512, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x3)
    x = BatchNormalization()(x)
    x = Convolution1D(512, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
    x4 = BatchNormalization()(x)

    x = Convolution1D(1024, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x4)
    x = BatchNormalization()(x)

    ## global_feature
    # global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = GlobalMaxPooling1D()(x)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([x, global_feature])
    c = Convolution1D(512, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    outputs = Convolution1D(num_classes, kernel_size=kernel_size, strides=1, padding='same', activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def mPointNet112(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    inputs = Input(shape=(num_points, 3))

    x = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(inputs)
    x1 = BatchNormalization()(x)
    # x = Convolution1D(64, 1, activation='relu')(x)
    # x1 = BatchNormalization()(x)

    x = Convolution1D(128, 1, activation='relu')(x1)
    x2 = BatchNormalization()(x)
    # x = Convolution1D(128, 1, activation='relu')(x)
    # x2 = BatchNormalization()(x)

    x = Convolution1D(256, 1, activation='relu')(x2)
    x3 = BatchNormalization()(x)
    # x = Convolution1D(256, 1, activation='relu')(x)
    # x3 = BatchNormalization()(x)

    x = Convolution1D(512, 1, activation='relu')(x3)
    x4 = BatchNormalization()(x)
    # x = Convolution1D(512, 1, activation='relu')(x)
    # x4 = BatchNormalization()(x)

    x = Convolution1D(1024, 1, activation='relu')(x4)
    x5 = BatchNormalization()(x)

    x = Convolution1D(2048, 1, activation='relu')(x5)
    x = BatchNormalization()(x)

    ## global_feature
    # global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = GlobalMaxPooling1D()(x)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([x, global_feature])
    c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    outputs = Convolution1D(num_classes, 1, activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def mPointNet113(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    inputs = Input(shape=(num_points, 3))

    x = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(inputs)
    x1 = BatchNormalization()(x)
    # x = Convolution1D(64, 1, activation='relu')(x)
    # x1 = BatchNormalization()(x)

    x = Convolution1D(128, 1, activation='relu')(x1)
    x2 = BatchNormalization()(x)
    # x = Convolution1D(128, 1, activation='relu')(x)
    # x2 = BatchNormalization()(x)

    x = Convolution1D(256, 1, activation='relu')(x2)
    x3 = BatchNormalization()(x)
    # x = Convolution1D(256, 1, activation='relu')(x)
    # x3 = BatchNormalization()(x)

    x = Convolution1D(512, 1, activation='relu')(x3)
    x4 = BatchNormalization()(x)
    # x = Convolution1D(512, 1, activation='relu')(x)
    # x4 = BatchNormalization()(x)

    x = Convolution1D(1024, 1, activation='relu')(x4)
    x5 = BatchNormalization()(x)

    x = Convolution1D(2048, 1, activation='relu')(x5)
    x = BatchNormalization()(x)

    ## global_feature
    # global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = GlobalMaxPooling1D()(x)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    ## point_net_seg
    c = concatenate([x, global_feature])
    c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    outputs = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def mPointNet114(num_points, num_classes):
    '''
        inputs:
            num_points: integer > 0, number of points for each point cloud image
            num_classes: total numbers of segmented classes
        outputs:
            onehot encoded array of classified points
    '''
    '''
    Begin defining Pointnet Architecture
    '''
    inputs = Input(shape=(num_points, 3))

    x = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(inputs)
    x1 = BatchNormalization()(x)

    x = Convolution1D(128, 1, activation='relu')(x1)
    x2 = BatchNormalization()(x)

    x = Convolution1D(256, 1, activation='relu')(x2)
    x3 = BatchNormalization()(x)

    x = Convolution1D(512, 1, activation='relu')(x3)
    x4 = BatchNormalization()(x)

    ## global_feature
    global_feature = GlobalMaxPooling1D()(x)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

    tx = concatenate([x4, global_feature])
    x = Convolution1D(128, 1, activation='relu')(tx)
    x1 = BatchNormalization()(x)

    x = Convolution1D(256, 1, activation='relu')(x1)
    x2 = BatchNormalization()(x)

    x = Convolution1D(512, 1, activation='relu')(x2)
    x3 = BatchNormalization()(x)

    x = Convolution1D(1024, 1, activation='relu')(x3)
    x4 = BatchNormalization()(x)

    ## global_feature
    global_feature = GlobalMaxPooling1D()(x)
    global_feature = Lambda(exp_dim0, arguments={'axis': 1})(global_feature)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)


    # x = Convolution1D(2048, 1, activation='relu')(x5)
    # x = BatchNormalization()(x)

    ## point_net_seg
    c = concatenate([x4, global_feature])
    c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    outputs = Convolution1D(num_classes, 1, activation='softmax')(c)
    ''' 
    End defining Pointnet Architecture
    '''

    ''' 
    Define Model
    '''
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def ReformationModule(inputs, num_points, num_filters, return_all=True):
    x = Convolution1D(num_filters, 1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(num_filters, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    if return_all:
        global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
        aggregated_feature = concatenate([x, global_feature])
        return aggregated_feature
    else:
        return global_feature

def RPointNetSeg1(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = ReformationModule(inputs=inputs, num_points=num_points, num_filters=16, return_all=True)
    x = ReformationModule(inputs=x, num_points=num_points, num_filters=64, return_all=True)
    x = ReformationModule(inputs=x, num_points=num_points, num_filters=256, return_all=True)

    c = Convolution1D(256, 1, padding='same', activation='relu')(x)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 1, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def RPointNetSeg13(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = Convolution1D(filters=4, kernel_size=num_points, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=8, kernel_size=num_points, padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=16, kernel_size=num_points, padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=32, kernel_size=num_points, padding='same')(x)
    x = BatchNormalization()(x)
    # x = Convolution1D(filters=64, kernel_size=num_points, padding='same')(x)
    # x = BatchNormalization()(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.summary())

    x, g = ReformationModule11(inputs=inputs, num_points=num_points, num_filters=4)
    g = Lambda(exp_dim, arguments={'num_points': num_points})(g)
    x = concatenate([x, g])
    x, g = ReformationModule11(inputs=x, num_points=num_points, num_filters=16)
    g = Lambda(exp_dim, arguments={'num_points': num_points})(g)
    x = concatenate([x, g])
    x, g = ReformationModule11(inputs=x, num_points=num_points, num_filters=64)
    g = Lambda(exp_dim, arguments={'num_points': num_points})(g)
    x = concatenate([x, g])
    x, g = ReformationModule11(inputs=x, num_points=num_points, num_filters=256)
    g = Lambda(exp_dim, arguments={'num_points': num_points})(g)
    x = concatenate([x, g])
    x, g = ReformationModule11(inputs=x, num_points=num_points, num_filters=512)
    g = Lambda(exp_dim, arguments={'num_points': num_points})(g)
    x = concatenate([x, g])

    c = Convolution1D(512, 3, padding='same', activation='relu')(x)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def RPointNetCls1(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = ReformationModule(inputs=inputs, num_points=num_points, num_filters=16, return_all=True)
    x = ReformationModule(inputs=x, num_points=num_points, num_filters=64, return_all=True)
    x = ReformationModule(inputs=x, num_points=num_points, num_filters=256, return_all=True)
    # x = ReformationModule(inputs=x, num_points=num_points, num_filters=1024, return_all=False)

    x = Convolution1D(1024, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    global_feature = MaxPooling1D(pool_size=num_points)(x)

    g = Flatten()(global_feature)

    c = Dense(units=256, activation='relu')(g)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.8)(c)
    # c = Dense(units=256, activation='relu')(c)
    # c = BatchNormalization()(c)
    # c = Dropout(rate=0.8)(c)
    # c = Dense(units=128, activation='relu')(c)
    # c = BatchNormalization()(c)

    outputs = Dense(units=num_classes, activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def ReformationModule11(inputs, num_points, num_filters):
    x = Convolution1D(filters=num_filters, kernel_size=1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=num_filters, kernel_size=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    g = MaxPooling1D(pool_size=num_points)(x)
    g = Lambda(exp_dim, arguments={'num_points': num_points})(g)
    x = concatenate([x, g])
    tx = Convolution1D(filters=num_filters, kernel_size=num_points, activation='relu', padding='same')(x)
    return tx

def RPointNetSeg11(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = Convolution1D(filters=8, kernel_size=1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)

    start_num_filters = 8
    x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    g1 = AveragePooling1D(pool_size=num_points)(x1)
    g1 = Lambda(exp_dim, arguments={'num_points': num_points})(g1)
    x1 = concatenate([x1, g1])
    x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Convolution1D(filters=start_num_filters, kernel_size=num_points, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)

    # model = Model(inputs=inputs, outputs=x1)
    # print(model.summary()) # 131

    x2 = Convolution1D(filters=start_num_filters*2, kernel_size=1, activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Convolution1D(filters=start_num_filters*2, kernel_size=1, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    g2 = AveragePooling1D(pool_size=num_points)(x2)
    g2 = Lambda(exp_dim, arguments={'num_points': num_points})(g2)
    x2 = concatenate([x2, g2])
    x2 = Convolution1D(filters=start_num_filters*2, kernel_size=1, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Convolution1D(filters=start_num_filters*2, kernel_size=int(num_points/2), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)

    # model = Model(inputs=inputs, outputs=x2)
    # print(model.summary()) #395

    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=1, activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=1, activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    g3 = AveragePooling1D(pool_size=num_points)(x3)
    g3 = Lambda(exp_dim, arguments={'num_points': num_points})(g3)
    x3 = concatenate([x3, g3])
    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=1, activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=int(num_points/4), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)

    # model = Model(inputs=inputs, outputs=x3)
    # print(model.summary()) # 923

    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=1, activation='relu', padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=1, activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    g4 = AveragePooling1D(pool_size=num_points)(x4)
    g4 = Lambda(exp_dim, arguments={'num_points': num_points})(g4)
    x4 = concatenate([x4, g4])
    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=1, activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=int(num_points / 8), activation='relu',
                       padding='same')(x4)
    x4 = BatchNormalization()(x4)

    # model = Model(inputs=inputs, outputs=x4)
    # print(model.summary()) # 1987

    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=1, activation='relu', padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=1, activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    g5 = AveragePooling1D(pool_size=num_points)(x5)
    g5 = Lambda(exp_dim, arguments={'num_points': num_points})(g5)
    x5 = concatenate([x5, g5])
    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=1, activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=int(num_points / 16), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)

    # model = Model(inputs=inputs, outputs=x5)
    # print(model.summary()) # 4144

    # x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x1)
    # x1 = BatchNormalization()(x1)
    # x2 = Convolution1D(filters=start_num_filters*2, kernel_size=1, activation='relu', padding='same')(x1)
    # x2 = BatchNormalization()(x2)
    # x3 = Convolution1D(filters=start_num_filters*4, kernel_size=1, activation='relu', padding='same')(x2)
    # x3 = BatchNormalization()(x3)
    # x4 = Convolution1D(filters=start_num_filters*8, kernel_size=1, activation='relu', padding='same')(x3)
    # x4 = BatchNormalization()(x4)
    # x5 = Convolution1D(filters=start_num_filters*16, kernel_size=1, activation='relu', padding='same')(x4)
    # x5 = BatchNormalization()(x5)
    # g = MaxPooling1D(pool_size=num_points)(x5)
    # g = Lambda(exp_dim, arguments={'num_points': num_points})(g)

    full_feat = concatenate([x1, x2, x3, x4, x5])
    c = Convolution1D(256, 3, padding='same', activation='relu')(full_feat)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def RPointNetCls11(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = Convolution1D(filters=8, kernel_size=1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)

    start_num_filters = 8
    x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    g1 = AveragePooling1D(pool_size=num_points)(x1)
    g1 = Lambda(exp_dim, arguments={'num_points': num_points})(g1)
    x1 = concatenate([x1, g1])
    x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Convolution1D(filters=start_num_filters, kernel_size=num_points, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)

    # model = Model(inputs=inputs, outputs=x1)
    # print(model.summary()) # 131

    x2 = Convolution1D(filters=start_num_filters * 2, kernel_size=1, activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Convolution1D(filters=start_num_filters * 2, kernel_size=1, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    g2 = AveragePooling1D(pool_size=num_points)(x2)
    g2 = Lambda(exp_dim, arguments={'num_points': num_points})(g2)
    x2 = concatenate([x2, g2])
    x2 = Convolution1D(filters=start_num_filters * 2, kernel_size=1, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Convolution1D(filters=start_num_filters * 2, kernel_size=int(num_points / 2), activation='relu',
                       padding='same')(x2)
    x2 = BatchNormalization()(x2)

    # model = Model(inputs=inputs, outputs=x2)
    # print(model.summary()) #395

    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=1, activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=1, activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    g3 = AveragePooling1D(pool_size=num_points)(x3)
    g3 = Lambda(exp_dim, arguments={'num_points': num_points})(g3)
    x3 = concatenate([x3, g3])
    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=1, activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Convolution1D(filters=start_num_filters * 4, kernel_size=int(num_points / 4), activation='relu',
                       padding='same')(x3)
    x3 = BatchNormalization()(x3)

    # model = Model(inputs=inputs, outputs=x3)
    # print(model.summary()) # 923

    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=1, activation='relu', padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=1, activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    g4 = AveragePooling1D(pool_size=num_points)(x4)
    g4 = Lambda(exp_dim, arguments={'num_points': num_points})(g4)
    x4 = concatenate([x4, g4])
    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=1, activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Convolution1D(filters=start_num_filters * 8, kernel_size=int(num_points / 8), activation='relu',
                       padding='same')(x4)
    x4 = BatchNormalization()(x4)

    # model = Model(inputs=inputs, outputs=x4)
    # print(model.summary()) # 1987

    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=1, activation='relu', padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=1, activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    g5 = AveragePooling1D(pool_size=num_points)(x5)
    g5 = Lambda(exp_dim, arguments={'num_points': num_points})(g5)
    x5 = concatenate([x5, g5])
    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=1, activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = Convolution1D(filters=start_num_filters * 16, kernel_size=int(num_points / 16), activation='relu',
                       padding='same')(x5)
    x5 = BatchNormalization()(x5)

    # model = Model(inputs=inputs, outputs=x5)
    # print(model.summary()) # 4144

    # x1 = Convolution1D(filters=start_num_filters, kernel_size=1, activation='relu', padding='same')(x1)
    # x1 = BatchNormalization()(x1)
    # x2 = Convolution1D(filters=start_num_filters*2, kernel_size=1, activation='relu', padding='same')(x1)
    # x2 = BatchNormalization()(x2)
    # x3 = Convolution1D(filters=start_num_filters*4, kernel_size=1, activation='relu', padding='same')(x2)
    # x3 = BatchNormalization()(x3)
    # x4 = Convolution1D(filters=start_num_filters*8, kernel_size=1, activation='relu', padding='same')(x3)
    # x4 = BatchNormalization()(x4)
    # x5 = Convolution1D(filters=start_num_filters*16, kernel_size=1, activation='relu', padding='same')(x4)
    # x5 = BatchNormalization()(x5)
    # g = MaxPooling1D(pool_size=num_points)(x5)
    # g = Lambda(exp_dim, arguments={'num_points': num_points})(g)

    full_feat = concatenate([x1, x2, x3, x4, x5])
    c = Convolution1D(128, 3, padding='same', activation='relu')(full_feat)
    c = BatchNormalization()(c)
    c = Convolution1D(1, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    c = Flatten()(c)
    c = Dense(units=512, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.25)(c)
    c = Dense(units=128, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.25)(c)

    outputs = Dense(units=num_classes, activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def ReformationModule2(inputs, input_size):
    num_points, num_channels = input_size
    x = Convolution1D(num_channels, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(num_channels, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    aggregated_feature = concatenate([x, global_feature])
    x = Convolution1D(num_channels, 3, activation='relu', padding='same')(aggregated_feature)
    x = BatchNormalization()(x)
    return x

def RPointNetSeg2(input_shape, num_classes):
    num_points, num_channels = input_shape

    inputs = Input(shape=input_shape)

    x = Convolution1D(8, kernel_size=3, padding='same', activation='relu')(inputs)
    num_channels = 8

    for i in range(6):
        x = ReformationModule2(inputs=x, input_size=(num_points, num_channels))
        x = ReformationModule2(inputs=x, input_size=(num_points, num_channels))
        x = Convolution1D(num_channels * 2, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        num_channels *= 2

    x = Convolution1D(int(num_channels/2), kernel_size=3, padding='same', activation='relu')(x)
    x = Convolution1D(int(num_channels/4), kernel_size=3, padding='same', activation='relu')(x)
    outputs = Convolution1D(filters=num_classes, kernel_size=3, padding='same', activation='relu')(x)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def RPointNetCls2(input_shape, num_classes):
    num_points, num_channels = input_shape

    inputs = Input(shape=input_shape)

    x = Convolution1D(8, kernel_size=3, padding='same', activation='relu')(inputs)
    num_channels = 8

    for i in range(6):
        x = ReformationModule2(inputs=x, input_size=(num_points, num_channels))
        x = ReformationModule2(inputs=x, input_size=(num_points, num_channels))
        x = Convolution1D(num_channels * 2, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        num_channels *= 2

    # c = Convolution1D(num_channels * 2, kernel_size=3, padding='same', activation='relu')(x)
    # c = BatchNormalization()(c)
    c = GlobalMaxPooling1D()(x)

    c = Dense(units=256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.25)(c)
    c = Dense(units=128, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.25)(c)

    outputs = Dense(units=num_classes, activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def mPointNet10(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = Convolution1D(16, 1, activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(16, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(64, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(64, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(256, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(256, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    c = Convolution1D(256, 1, padding='same', activation='relu')(x)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 1, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def mPointNet11(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = Convolution1D(16, 3, activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    c = Convolution1D(256, 3, padding='same', activation='relu')(x)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def mPointNet12(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    x = Convolution1D(16, 3, activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    x = Convolution1D(16, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(64, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(256, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    x = Convolution1D(1024, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    global_feature = MaxPooling1D(pool_size=num_points)(x)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([x, global_feature])

    c = Convolution1D(1024, 3, padding='same', activation='relu')(x)
    c = BatchNormalization()(c)
    c = Convolution1D(512, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model

def mPointNet3(num_points, num_classes):
    input_shape = (num_points, 3)

    inputs = Input(shape=input_shape)

    conv11 = Convolution1D(16, 3, activation='relu', padding='same', input_shape=input_shape)(inputs)
    bn11 = BatchNormalization()(conv11)
    conv12 = Convolution1D(16, 1, activation='relu', padding='same')(bn11)
    bn12 = BatchNormalization()(conv12)

    ft = MaxPooling1D(pool_size=num_points)(bn12)
    ft = Dense(256, activation='relu')(ft)
    ft = BatchNormalization()(ft)
    ft = Dense(16*16, weights=[np.zeros([256, 16 * 16]), np.eye(16).flatten().astype(np.float32)])(ft)
    ft_mt = Reshape((16, 16))(ft)
    nft = keras.layers.dot(inputs=[bn12, ft_mt], axes=2)

    global_feature = MaxPooling1D(pool_size=num_points)(nft)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([bn11, bn12, nft, global_feature])

    conv21 = Convolution1D(64, 3, padding='same', activation='relu')(x)
    bn21 = BatchNormalization()(conv21)
    conv22 = Convolution1D(64, 1, padding='same', activation='relu')(bn21)
    bn22 = BatchNormalization()(conv22)

    ft = MaxPooling1D(pool_size=num_points)(bn22)
    ft = Dense(256, activation='relu')(ft)
    ft = BatchNormalization()(ft)
    ft = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(ft)
    ft_mt = Reshape((64, 64))(ft)
    nft = keras.layers.dot(inputs=[bn22, ft_mt], axes=2)

    global_feature = MaxPooling1D(pool_size=num_points)(bn22)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([bn21, bn22, nft, global_feature])

    conv31 = Convolution1D(256, 3, padding='same', activation='relu')(x)
    bn31 = BatchNormalization()(conv31)
    conv32 = Convolution1D(256, 1, padding='same', activation='relu')(bn31)
    bn32 = BatchNormalization()(conv32)

    global_feature = MaxPooling1D(pool_size=num_points)(bn32)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    x = concatenate([bn31, bn32, global_feature])

    c = Convolution1D(512, 3, padding='same', activation='relu')(x)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 3, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)

    outputs = Convolution1D(num_classes, 3, padding='same', activation='softmax')(c)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model


from keras import backend as K

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is PointNet Definition Program')

    if argv is None:
        argv = sys.argv

    if len(argv) > 1:
        for i in range(len(argv) - 1):
            print(argv[i + 1])


if __name__ == '__main__':
    main()
