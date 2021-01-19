# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:09:16 2020

@author: pc
"""

import tensorflow as tf
import unet
import numpy as np
import support_based as spb

class UNET:
    def __init__(self, i_w, i_h, i_ch, o_w, o_h, o_cl, lean_rate, sa_str, train_flag, GPU_str = 'no'):
        # clean graph
        tf.reset_default_graph()
        # data parameters
        self.image_w = i_w
        self.image_h = i_h
        self.image_c = i_ch
        self.mask_w = o_w
        self.mask_h = o_h
        self.class_num = o_cl
        self.learning_rate = lean_rate
        
        # placeholder
        self.input_data = tf.placeholder(tf.float32, [None, self.image_w, self.image_h, self.image_c])
        self.output_mask = tf.placeholder(tf.float32, [None, self.mask_w, self.mask_h, self.class_num])
        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        
        # whether GPU
        self.GPU_str = GPU_str
        
        # whether training
        self.tr_flag = train_flag
        self.model_path = spb.mo_path + 'save_results/' + sa_str + '/model/model.ckpt'
        self.sa_str = sa_str
        
    def build_framework(self):
        # Unet
        self.output = unet.Unet(name="UNet", in_data = self.input_data, width = self.image_w,
                                height = self.image_h, channel = self.image_c, cal_num = self.class_num,
                                is_train = self.is_train, reuse = False)
        # loss
        self.loss = unet.uu.generalized_dice_loss(self.output_mask, self.output)
        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, name="opt")
            
        self.init = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        
        if self.GPU_str == 'yes':
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        else:
            self.sess = tf.Session()
        
        # whether training the model or load trained model
        if self.tr_flag == 'yes':
            self.sess.run(self.init)
        else:
            self.saver.restore(self.sess, self.model_path)
                
        
    # train
    def train(self, input_data, output_data):
        optim, tr_loss, tr_out = self.sess.run([self.opt, self.loss, self.output], feed_dict={self.input_data: input_data,
                                                self.output_mask: output_data, self.is_train: True, self.lr: self.learning_rate})
        return tr_loss, tr_out
    

    # test
    def test(self, da, b_num):
        los = []
        pred_li = []
        for ind in range(10000):
            bat_img, bat_mask = da.get_test_bat_img()      
            te_loss, te_out = self.sess.run([self.loss, self.output], feed_dict={self.input_data: bat_img,
                                                self.output_mask: bat_mask, self.is_train: False, self.lr: self.learning_rate})
            los.append([te_loss])
            pred_li.append(te_out)

            # check whether ended
            if da.te_ind == 0:
                break
        
        # calculate DICE
        pred_arr = np.concatenate(pred_li, 0)
        pred_mask = spb.pred_to_mask(pred_arr)
        Dice = spb.mean_dice(da.te_m, pred_mask)
        
        return np.concatenate(los), Dice, pred_arr
    
    # predict
    def pred(self, in_img):
        mask_li = []
        b_num = 10
        for ind in range(1000):
            st_num = (ind*b_num)%in_img.shape[0]
            bat_img = in_img[st_num: st_num + b_num]
            pre_out = self.sess.run([self.output], feed_dict={self.input_data: bat_img, self.is_train: False})
            mask_li.append(pre_out[0])
            
            if bat_img.shape[0] < b_num or (ind + 1)*b_num == in_img.shape[0]:
                break
            
        return np.concatenate(mask_li, 0)
     
    # save the model
    def save(self):
        self.saver.save(self.sess, self.model_path)
        
























