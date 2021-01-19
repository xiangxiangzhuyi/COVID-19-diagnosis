# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:09:20 2020

@author: pc
"""

import run
import argparse

# pass parameters to the script
pa = argparse.ArgumentParser(description='manual to this script')
pa.add_argument('--w_num', type=int, default = None)
pa.add_argument('--tr_bat', type=int, default = None)
pa.add_argument('--te_bat', type=int, default = None)
pa.add_argument('--aug_str', type=str, default = None)
pa.add_argument('--onl_lun', type=str, default = None)
pa.add_argument('--resi_str', type=str, default = None)
ar = pa.parse_args()

run.run_model(w_num = ar.w_num, h_num= 1, tr_bat= ar.tr_bat, te_bat = ar.te_bat, GPU_str='yes', 
              iter_num= 10000, test_once= 50, aug_str=ar.aug_str, aug_num = 5, onl_lun = ar.onl_lun, 
              resi_str = ar.resi_str)
