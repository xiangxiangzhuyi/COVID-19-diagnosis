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

def run_model(cat_num, cha_num, lean_rate, tr_fl, bat_num, va_ind, iter_num, va_once, va_tim, GPU_str, sa_str=None):
    # saving folder
    if tr_fl == 'yes':
        sa_str = spb.com_mul_str([cat_num, cha_num, va_ind, bat_num, va_tim, GPU_str])
        
    # construct dataset
    if va_ind != 5:
        tr_ind = list(range(5))
    else:
        tr_ind = list(range(6))
    tr_ind.remove(va_ind)
    da = spda.dataset(bat_num, [tr_ind, [va_ind], [va_ind]], 'yes')
    
    # construct UNET
    FCN = model.FCN_model(cat_num, cha_num, lean_rate, tr_fl,  GPU_str, sa_str)
    FCN.build_framework()

    # create lists to save the results
    tr_res = []
    va_res = []
    for i in range(iter_num):
        # train
        b_i, b_l, b_n = da.get_img('tr', 'yes')  
        tr_loss, tr_acc = FCN.train(b_i, b_l)
        tr_res.append(np.array([i, da.ind_fla['tr'], tr_loss, tr_acc]))
        
        print('train', i, tr_loss, tr_acc, time.ctime())
        
        # validation
        if (i+1)%va_once == 0:
            va_loss, va_acc = FCN.validate(da, va_tim, 'te')
            va_res.append(np.array([i, va_loss, va_acc]))
            
            # print
            print('test', i, va_loss, va_acc, time.ctime())
            
            # save result
            spb.save_result(np.stack(tr_res, 0), sa_str, 'tr_res.npy')
            spb.save_result(np.stack(va_res, 0), sa_str, 'va_res.npy')
            FCN.save()
            if va_acc > 0.91:
                break
    
    # output volume
    FCN.pred(da)
        

    
    
    
        
        
