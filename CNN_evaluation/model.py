# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:09:16 2020

@author: pc
"""

import tensorflow as tf
import numpy as np
import support_based as spb
import support_framework as spf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

class FCN_model:
    def __init__(self, cat_num, cha_num, lean_rate, tr_fl, kp, GPU_str, sa_str):
        # clean graph
        tf.reset_default_graph()
        # data parameters
        self.lr = lean_rate
        self.cat_num = cat_num
        self.cha_num = cha_num
        self.kp = kp
        self.GPU_str = GPU_str
        self.sa_str = sa_str
        
        # placeholder
        self.img = tf.placeholder(tf.float32, [None, 254, 373, 60, 1])
        self.lab = tf.placeholder(tf.int32, [None])
        self.is_tr = tf.placeholder(tf.bool)
        self.tf_lr = tf.placeholder(tf.float32)
        self.tf_kp = tf.placeholder(tf.float32)
        
        # whether training
        self.tr_fl = tr_fl
        self.model_path = spb.mo_path + 'save_results/' + sa_str + '/model/model.ckpt'
        self.sa_str = sa_str
        
    def build_framework(self):
        # FCN
        self.y_conv = spf.CNN('cnn', self.img, self.cat_num, self.cha_num, self.tf_kp, self.is_tr)
        
        # one hot
        self.one_hot = tf.one_hot(self.lab, self.cat_num)
        
        # loss
        self.loss = spf.ut.loss(self.y_conv, self.one_hot)
        
        # accuracy
        self.acc = spf.ut.acc(self.y_conv, self.one_hot)

        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="opt")
        
        # initinalize
        self.init = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        # whether GPU
        if self.GPU_str == 'yes':
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        else:
            self.sess = tf.Session()

        # whether training the model or load trained model
        if self.tr_fl == 'yes':
            self.sess.run(self.init)
        else:
            self.saver.restore(self.sess, self.model_path)
        
    # train
    def train(self, b_i, b_l):
        optim, tr_loss, tr_acc = self.sess.run([self.opt, self.loss, self.acc], feed_dict={self.img: b_i,
                                                self.lab: b_l, self.is_tr: True, self.tf_lr: self.lr,
                                                self.tf_kp: self.kp})
        return tr_loss, tr_acc
    
    # test
    def pred(self, da, da_ty):
        los_li, acc_li = [], []
        for ind in range(10000):
            b_i, b_l, b_n = da.get_img(da_ty, 'no')       
            va_loss, va_acc = self.sess.run([self.loss, self.acc], feed_dict={self.img: b_i,
                                                self.lab: b_l, self.is_tr: False, 
                                                self.tf_lr: self.lr, self.tf_kp: 1.})
            los_li.append(va_loss)
            acc_li.append(va_acc)
            
            # check whether ended
            if da.ind_fla[da_ty] == 0:
                break
            
        return np.mean(los_li), np.mean(acc_li)
    
    # save the model
    def save(self):
        self.saver.save(self.sess, self.model_path)
        
























