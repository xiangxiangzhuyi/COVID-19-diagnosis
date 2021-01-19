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
def CNN(name, in_data, cat_num, cha_num, kp, is_train, reuse = False):
    
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
        net = ut.conv_layer(net, [3, 3, 3, 8*cha_num, 8*cha_num], 1, 'SAME', is_train)
        net = ut.conv_layer(net, [3, 3, 3, 8*cha_num, 16*cha_num], 1, 'SAME', is_train)
        net = ut.max_pools3d(net, 2, 2, 'SAME')
        
        # layer 5
        net = ut.conv_layer(net, [3, 3, 3, 16*cha_num, 16*cha_num], 1, 'SAME', is_train)
        net = ut.conv_layer(net, [3, 3, 3, 16*cha_num, 32*cha_num], 1, 'SAME', is_train)
        net = ut.max_pools3d(net, 2, 2, 'SAME')
        
        # average pool
        net = ut.avg_pools3d(net, 8, 12, 2, 1)
        net = net[:,0,0,0,:]
        
        # MLP layer 1
        w1 = ut.weight_variable([32*cha_num, 200])
        net = tf.matmul(net, w1)
        net = tf.layers.batch_normalization(net, training = is_train)
        net = tf.nn.leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob = kp)
        
        # MLP layer 2
        w2 = ut.weight_variable([200, 100])
        net = tf.matmul(net, w2)
        net = tf.layers.batch_normalization(net, training = is_train)
        net = tf.nn.leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob = kp)
        
        # softmax layer
        w3 = ut.weight_variable([100, cat_num])
        b3 = ut.bias_variable([cat_num])
        y_conv = tf.nn.softmax(tf.matmul(net, w3) + b3)
        
        return y_conv

def MLP(name, in_da, cat_num, kp, is_train, reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        # layer 1
        w1 = ut.weight_variable([in_da.shape.as_list()[-1], 200])
        net = tf.matmul(in_da, w1)
        net = tf.layers.batch_normalization(net, training = is_train)
        net = tf.nn.leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob = kp)
        
        # layer
        w2 = ut.weight_variable([200, 100])
        net = tf.matmul(net, w2)
        net = tf.layers.batch_normalization(net, training = is_train)
        net = tf.nn.leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob = kp)
        
        # layer 2
        w5 = ut.weight_variable([100, cat_num])
        b2 = ut.bias_variable([cat_num])
        y_conv = tf.nn.softmax(tf.matmul(net, w5) + b2)

    return y_conv
        
