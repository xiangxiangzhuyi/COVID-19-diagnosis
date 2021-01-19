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
pa.add_argument('--cha_num', type=int, default = None)
pa.add_argument('--bat_num', type=int, default = None)
pa.add_argument('--iter_num', type=int, default = None)
pa.add_argument('--va_once', type=int, default = None)
pa.add_argument('--va_tim', type=int, default = None)
pa.add_argument('--va_ind', type=int, default = None)
pa.add_argument('--tr_fl', type=str, default = None)
pa.add_argument('--sa_str', type=str, default = None)
ar = pa.parse_args()

run.run_model(cat_num = 3, lean_rate = 0.0005, GPU_str = 'yes', 
              bat_num = ar.bat_num, cha_num = ar.cha_num, iter_num = ar.iter_num, 
              va_once = ar.va_once, va_tim = ar.va_tim, tr_fl = ar.tr_fl, 
              sa_str = ar.sa_str, va_ind = ar.va_ind)