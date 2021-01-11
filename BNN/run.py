# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:06:59 2020

@author: pc
"""

import model
import numpy as np
import support_based as spb
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def run_model(tr_fl, va_fl, te_fl, cat_num, r_num, thre, iter_num, va_once, sa_path):
    
    # set the configuration of the BNN
    cat_num = np.max(tr_fl[:, -1]) +1
    fea_num = tr_fl.shape[-1] -1
    sam_num = tr_fl.shape[0]
    bat_num = 50
    
    # construct UNET
    BNN = model.BNN_model(cat_num, fea_num, sam_num, r_num, 0.01, 0.5, 'yes')
    BNN.build_framework()

    # create lists to save the results
    ind = 0
    max_acc = 0
    for i in range(iter_num):
        # train
        b_in, ind = spb.get_bat_index(np.arange(tr_fl.shape[0]), bat_num, ind)
        b_fl = tr_fl[b_in]
        tr_loss, tr_acc = BNN.train(b_fl[:, 0:-1], b_fl[:, -1])
        
        print('train', i, tr_loss, tr_acc)
        
        # validation
        if (i+1)%va_once == 0:
            va_loss, va_acc = BNN.test(va_fl[:, 0:-1], va_fl[:, -1])
            
            # print
            print('test', i, va_loss, va_acc)
            if va_acc >= max_acc:
                max_acc = va_acc
            if va_acc > thre - 0.01:
                break

    # uncer
    tr_p = BNN.cal_uncer(tr_fl[:, 0:-1], tr_fl[:, -1])
    va_p= BNN.cal_uncer(va_fl[:, 0:-1], va_fl[:, -1])
    te_p= BNN.cal_uncer(te_fl[:, 0:-1], te_fl[:, -1])
   
    # save result
    np.save(sa_path + '/tr_p.npy', tr_p)
    np.save(sa_path + '/va_p.npy', va_p)
    np.save(sa_path + '/te_p.npy', te_p)
    
    return max_acc
        

    
    
    
        
        
