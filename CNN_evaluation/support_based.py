# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:43:45 2020

@author: pc
"""

import numpy as np
import pickle
import random
import string
import os
import matplotlib.pyplot as plt
import cv2


# set path
if os.path.exists('E:/Project15_COVID19_CT/') :
    pro_path = 'E:/Project15_COVID19_CT/'
else:
    pro_path = '/project/huangli/Project/Project15_COVID19_CT/'
mo_path = pro_path + 'Models/model18_3D_volume_final_experiment_Critical_illness/'

import sys
sys.path.insert(1, pro_path + 'Models/utiles/')
import read_all_img_general as raig


# convert to one hot images
def conv_one_hot(img, cal_num):
    mat_li = []
    for i in range(int(cal_num)):
        ma = img.copy()
        ma[np.where(ma == i)] = 100
        ma[np.where(ma != 100)] = 0
        ma[np.where(ma == 100)] = 1
        mat_li.append(ma)
        
    return np.stack(mat_li, -1)

def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''
    
    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1: ]    


# save the model result
def save_result(result, strs, fi):
    # create folder
    folder_name = mo_path + 'save_results/' + strs + '/'
    if  not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # file name list
    np.save(folder_name + fi, result)   
        
    return

# read the model result
def read_result(folder, file):
    # folder name
    file_full_name = mo_path + 'save_results/' + folder + '/' + file
    if os.path.isfile(file_full_name):
        data = np.load(file_full_name, encoding='bytes', allow_pickle=True)
    else:
        data = 'nofile'
    
    return data

# save list
def save_list(result, strs, fi):
    folder_name = mo_path + 'save_results/' + strs + '/'
    if  not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # file name list
    with open(folder_name + fi, 'wb') as file:
        pickle.dump(result, file)
        
    return

# read list
def read_list(strs, fi):
    file_full_name = mo_path + 'save_results/' + strs + '/' + fi
    if os.path.isfile(file_full_name):
        with open(file_full_name, 'rb') as file:
            res = pickle.load(file)
    else:
        res = 'nofile'
    return res

# read result
def read_tr_va_res(folder):
    tr_res = read_result(folder, 'tr_res.npy')
    va_res = read_result(folder, 'va_res.npy')
    te_res = read_result(folder, 'te_res.npy')
    
    max_va_acc = np.max(va_res[:, -1])
    min_va_los = np.min(va_res[:, -2])
    
    return tr_res, va_res, te_res,  max_va_acc, min_va_los
