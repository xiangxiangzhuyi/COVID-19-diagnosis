# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:06:59 2020

@author: pc
"""

import model
import support_datasets as spda
import numpy as np
import support_based as spb
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def run_model(cat_num, cha_num, lean_rate, kp, tr_fl, bat_num, iter_num, va_once, GPU_str, sa_str=None):
    # saving folder
    if tr_fl == 'yes':
        sa_str = spb.com_mul_str([cat_num, cha_num, bat_num, GPU_str])
        
    # construct dataset
    # construct dataset
    da = spda.dataset(bat_num, [[0,1,2,3,4], [5], [5]], 'yes')
    
    # construct UNET
    FCN = model.FCN_model(cat_num, cha_num, lean_rate, tr_fl, kp, GPU_str, sa_str)
    FCN.build_framework()

    # create lists to save the results
    tr_res = []
    va_res = []
    for i in range(iter_num):
        # train
        b_i, b_l, b_n = da.get_img('tr', 'no')  
        tr_loss, tr_acc = FCN.train(b_i, b_l)
        tr_res.append(np.array([i, da.ind_fla['tr'], tr_loss, tr_acc]))
        
        print('train', i, tr_loss, tr_acc, time.ctime())
        
        # validation
        if (i+1)%va_once == 0:
            va_loss, va_acc = FCN.pred(da, 'va')
            va_res.append(np.array([i, va_loss, va_acc]))
            
            # print
            print('test', i, va_loss, va_acc, time.ctime())
            
            # save result
            spb.save_result(np.stack(tr_res, 0), sa_str, 'tr_res.npy')
            spb.save_result(np.stack(va_res, 0), sa_str, 'va_res.npy')
            FCN.save()
            if va_acc > 0.95:
                break
    
    # test
    te_loss, te_acc = FCN.pred(da, 'te')
   
    # print
    print('test', i, te_loss, te_acc, time.ctime())
    
    # save result
    spb.save_result(np.array([te_loss, te_acc]), sa_str, 'te_res.npy')
        

    
    
    
        
        
