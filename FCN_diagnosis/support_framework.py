# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:33:43 2020

@author: pc
"""
# this script is used to construct the framework of U-net


# import package
import tensorflow as tf
import utils as ut

# the framework is the same as brain paper
def FCN(name, in_data, cat_num, cha_num, is_train, reuse = False):
    
    # construction
    with tf.variable_scope(name, reuse = reuse):
        # layer 1 
        net = ut.conv_layer(in_data, [3, 3, 3, 1, cha_num], 1, 'SAME', is_train)
        net = ut.conv_layer(net, [3, 3, 3, cha_num, 2*cha_num], 1, 'SAME', is_train)
        net = ut.max_pools3d(net, 2, 2, 'SAME')
        
        # layer 2
        net = ut.conv_layer(net, [3, 3, 3, 2*cha_num, 2*cha_num], 1, 'SAME', is_train)
        net = ut.conv_layer(net, [3, 3, 3, 2*cha_num, 4*cha_num], 1, 'SAME', is_train)
        net = ut.max_pools3d(net, 2, 2, 'SAME')
        
        # layer 3
        net = ut.conv_layer(net, [3, 3, 3, 4*cha_num, 4*cha_num], 1, 'SAME', is_train)
        net = ut.conv_layer(net, [3, 3, 3, 4*cha_num, 8*cha_num], 1, 'SAME', is_train)
        net = ut.max_pools3d(net, 2, 2, 'SAME')
        
        # layer 4
        w4 = ut.weight_variable([6, 6, 6, 8*cha_num, cat_num])
        net = ut.conv3d(net, w4, 1, 'VALID')
        net = tf.layers.batch_normalization(net, training = is_train)
        net = tf.nn.relu(net)
        
        # softmax
        net = tf.nn.softmax(net)
        
        return net
        
def FCN_ours(name, in_data, cat_num, is_train, kp1, kp2, reuse = False):
    # construction
    with tf.variable_scope(name, reuse = reuse):
    
    
    
        return 