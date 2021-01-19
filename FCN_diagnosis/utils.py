# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:35:54 2020

@author: pc
"""
# This script incloud some convolutional functions

import tensorflow as tf
import math

# convolutional functions ------------------------------------------------------------------------
def conv_layer(inp, shape, stride, pa_str, is_train):
    w = weight_variable(shape)
    inp = conv3d(inp, w, stride, pa_str)
    inp = tf.layers.batch_normalization(inp, training = is_train)
    outp = tf.nn.relu(inp)
    return outp
   
def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def conv3d(a, w, s, pa_str):
    return tf.nn.conv3d(a, w, strides=[1, s, s, s, 1], padding= pa_str)
    
def max_pools3d(a, h, s, pa_str):
    return tf.nn.max_pool3d(a, ksize=[1, h, h, h, 1], strides=[1, s, s, s, 1], padding = pa_str)

def loss(pred_y, onehot):
    diff = tf.losses.softmax_cross_entropy(onehot_labels = onehot, logits = pred_y)
    return tf.reduce_mean(diff)

# accuracy
def acc(pred_y, onehot):
    cor_pre = tf.equal(tf.argmax(pred_y, 1), tf.argmax(onehot, 1))
    return tf.reduce_mean(tf.cast(cor_pre, "float"))





