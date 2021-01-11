# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:35:23 2020

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:36:42 2020

@author: DELL
"""

import support_based as spb
import run
import argparse

# pass parameters to the script
pa = argparse.ArgumentParser(description='manual to this script')
pa.add_argument('--num', type=int, default = 0)
ar = pa.parse_args()

# read the dataset
fo_li = [['model29_classify', 'validation']]

def uncer(fl):
    max_acc = run.run_model(tr_fl = fl[1][0], va_fl = fl[1][1], te_fl = fl[1][2], cat_num = 2, r_num= 100, 
              iter_num= 10000, va_once=50, thre =1., sa_path = fl[0])
    
    max_acc = run.run_model(tr_fl = fl[1][0], va_fl = fl[1][1], te_fl = fl[1][2], cat_num = 2, r_num= 100, 
              iter_num= 50000, va_once=50, thre = max_acc, sa_path = fl[0])
    return 

num = ar.num
all_fea = spb.read_fea_na(fo_li[num][0], fo_li[num][1])
#[uncer(fea) for fea in all_fea]

