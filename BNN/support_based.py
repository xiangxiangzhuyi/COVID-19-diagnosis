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
mo_path = pro_path + 'Models/model21_BNN_test/'

import sys
sys.path.insert(1, pro_path + 'Models/utiles/')

# read feature
def read_fea(mo_na):
    fu_sa = pro_path + 'Models/' + mo_na + '/save_results/'
    sa_d = os.listdir(fu_sa)
    
    # traverse
    all_fea = []
    fe_na = ['tr_fl.npy', 'va_fl.npy', 'te_fl.npy']
    for ca in sa_d:
        fu_ca = fu_sa + ca + '/'
        ca_d = os.listdir(fu_ca)
        
        # save model list
        for sm in ca_d:
            fu_sm = fu_ca + sm
            if '0' in sm and os.path.isdir(fu_sm):
                fea_li = [np.load(fu_sm + '/' + x) for x in fe_na]
                all_fea.append([fu_sm, fea_li])
    
    return all_fea


# read by the save folder name
def read_fea_na(mo_na, sa_na):
    fu_ca = pro_path + 'Models/' + mo_na + '/save_results/' + sa_na + '/'
    ca_d = os.listdir(fu_ca)
    
    all_fea = []
    fe_na = ['tr_fl.npy', 'va_fl.npy', 'te_fl.npy']
    
    for sm in ca_d:
        fu_sm = fu_ca + sm
        if '0' in sm and os.path.isdir(fu_sm):
            fea_li = [np.load(fu_sm + '/' + x) for x in fe_na]
            
            # convert 3 classes problem to 2 classes problem
            if np.max(fea_li[0][:, -1]) == 2:
                for x in range(len(fea_li)):
                    lab = fea_li[x][:, -1]
                    lab[lab < 2] = 0
                    lab[lab == 2] = 1
                    fea_li[x][:, -1] = lab
            
            all_fea.append([fu_sm, fea_li])

    return all_fea


def get_bat_index(index_list, bat_num, ind):
    if bat_num >= index_list.shape[0]:
        bat_index = index_list
        ind = 0
    else:
        st_num = (ind*bat_num)%len(index_list)
        bat_index = index_list[st_num: st_num + bat_num]
        if bat_index.shape[0] < bat_num:
            ind = 0
        elif bat_index.shape[0] == bat_num and (ind + 1)*bat_num == len(index_list):
            ind = 0    
        else:
            ind = ind + 1
    return bat_index, ind


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
    
    max_va_acc = np.max(va_res[:, -1])
    min_va_los = np.min(va_res[:, -2])
    
    return tr_res, va_res, max_va_acc, min_va_los
