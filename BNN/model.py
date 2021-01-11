# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:09:16 2020

@author: pc
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np
import warnings
# warnings.simplefilter(action="ignore")
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# Dependency imports


class BNN_model:
    def __init__(self, cat_num, fea_num, sam_num, r_num, lean_rate, dp, GPU_str):
        # clean graph
        tf.reset_default_graph()
        # clean graph
        tf.reset_default_graph()
        # data parameters
        self.cat_num = int(cat_num)
        self.fea_num = fea_num
        self.sam_num = sam_num
        self.r_num = int(r_num)
        self.lr = lean_rate
        self.dp = dp
        self.GPU_str = GPU_str
        
        # placeholder
        self.fea = tf.placeholder(tf.float32, shape=[None, fea_num])
        self.lab = tf.placeholder(tf.int64, shape=[None])
        self.tf_dp = tf.placeholder(tf.float32)
        
    def build_framework(self):
        # define the model
        self.neural_net = tf.keras.Sequential([
              tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
              tf.keras.layers.Dropout(self.tf_dp),
              tfp.layers.DenseFlipout(128, activation=tf.nn.relu),
              tf.keras.layers.Dropout(self.tf_dp),
              tfp.layers.DenseFlipout(self.cat_num)])
        
        self.logits = self.neural_net(self.fea)
        self.lab_d = tfp.distributions.Categorical(logits = self.logits)
 
        # loss
        self.nll = -tf.reduce_mean(self.lab_d.log_prob(self.lab))
        self.kl = sum(self.neural_net.losses) / self.sam_num
        self.elbo_loss = self.nll + self.kl
        
        # optimize
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.tr_op = optimizer.minimize(self.elbo_loss)
        
        # accuracy
        self.pred = tf.argmax(self.logits, axis = 1)
        cor_pre = tf.equal(self.pred, self.lab)
        self.acc = tf.reduce_mean(tf.cast(cor_pre, "float"))
        
        # initialization
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # whether GPU
        if self.GPU_str == 'yes':
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        else:
            self.sess = tf.Session()

        # run
        self.sess.run(self.init_op)
        
    # train
    def train(self, b_f, b_l):
        _, tr_loss, tr_acc = self.sess.run([self.tr_op, self.elbo_loss, self.acc], feed_dict={self.fea: b_f,
                                                self.lab: b_l, self.tf_dp: self.dp})
        return tr_loss, tr_acc
    
    
    def test(self, b_f, b_l):
        te_loss, te_acc = self.sess.run([self.elbo_loss, self.acc], feed_dict={self.fea: b_f,
                                                self.lab: b_l, self.tf_dp: self.dp})
        return te_loss, te_acc
    
    # get the prob distribution
    def cal_uncer(self, b_f, b_l):
        probs = np.asarray([self.sess.run(self.lab_d.probs, feed_dict={self.fea: b_f, self.lab: b_l, self.tf_dp: 0.}) for _ in  range(self.r_num)])
    
        return probs























