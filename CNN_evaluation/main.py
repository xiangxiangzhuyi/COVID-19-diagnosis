# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:09:20 2020

@author: pc
"""

import run
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import argparse

# pass parameters to the script
pa = argparse.ArgumentParser(description='manual to this script')
pa.add_argument('--va_ind', type=int, default = None)
pa.add_argument('--cha_num', type=int, default = None)
ar = pa.parse_args()

run.run_model(cat_num = 2, cha_num = ar.cha_num, lean_rate = 0.0005, kp=0.9, bat_num = 10, 
              iter_num = 20000, va_once = 100, tr_fl = 'yes', GPU_str = 'yes', va_ind = ar.va_ind)