# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:35:01 2020

@author: DELL
"""

import support_based as spb
import pickle
import numpy as np
import os

# get best point locations
def get_loca(folder, sel_num):
    with open(spb.mo_path + 'stat_res/divid_da', 'rb') as file:
        d_li = pickle.load(file)
    
    # assign parts into training, validation, and test
    str_li = folder.split('_')
    if len(str_li) == 7:
        va_ind = int(str_li[2])
    else:
        va_ind = 5
        
    if va_ind != 5:
        tr_ind = list(range(5))
    else:
        tr_ind = list(range(6))
    tr_ind.remove(va_ind)
    da_inf = [np.concatenate([d_li[y] for y in x]) for x in [tr_ind, [va_ind], [va_ind]]]
        
    sa_fo = 'save_results/' + folder + '/volume/'
    
    # get the shape of the probability map
    s_fi = spb.mo_path + sa_fo
    a, b, c, d = np.load(s_fi + os.listdir(s_fi)[0]).shape
    
    # traverse all file
    cat_arr = np.zeros((a, b, c, d))
    all_arr = np.zeros((a, b, c, d))
    for i in range(da_inf[0].shape[0]):
        # read
        fu_fo = spb.mo_path + sa_fo + da_inf[0][i, 0] + '.npy'
        if not os.path.isfile(fu_fo):
            continue
        pm = np.load(fu_fo)
        pl = np.argmax(pm, axis= -1)
        gr = int(float(da_inf[0][i, 8]))
        
        # analysis
        pl[pl != gr] = 0
        pl[pl == gr] = 1
        cat_arr[:,:,:, gr] = cat_arr[:,:,:, gr] + pl
        all_arr[:,:,:, gr] = all_arr[:,:,:, gr] + 1
    
    # summary
    cat_acc = cat_arr/all_arr
    all_acc = np.sum(cat_arr, -1)/np.sum(all_arr, -1)
    
    # select
    cat_loca = [sel(cat_acc[:,:,:, x], sel_num, a,b,c) for x in range(cat_acc.shape[-1])]
    all_loca = sel(all_acc, sel_num, a,b,c)

    return cat_loca, all_loca

def sel(acc, sel_num, a, b, c):
    ind = np.argsort(np.reshape(acc, (a*b*c)))
    ind = np.reshape(ind, (a, b, c))
    loca = [np.where(ind == x) for x in range(sel_num)]
    
    return loca
    
# extract features
def extract_fea(folder, loca, fea_ty):
    with open(spb.mo_path + 'stat_res/divid_da', 'rb') as file:
        d_li = pickle.load(file)
    
    # assign parts into training, validation, and test
    str_li = folder.split('_')
    if len(str_li) == 7:
        va_ind = int(str_li[2])
    else:
        va_ind = 5
        
    if va_ind != 5:
        tr_ind = list(range(5))
    else:
        tr_ind = list(range(6))
    tr_ind.remove(va_ind)
    da_inf = [np.concatenate([d_li[y] for y in x]) for x in [tr_ind, [va_ind], [va_ind]]]
    
    
    sa_fo = 'save_results/' + folder + '/volume/'
    # traverse all file
    all_fea_lab = []
    for da in da_inf:
        fea_lab_li = []
        for i in range(da.shape[0]):
            # read
            fu_fo = spb.mo_path + sa_fo + da[i, 0] + '.npy'
            if not os.path.isfile(fu_fo):
                continue
            
            pm = np.load(fu_fo)
            
            # extract feature
            pl = np.argmax(pm, axis= -1)
            if fea_ty == 'int':
                fea = np.concatenate([pl[x] for x in loca])
            elif fea_ty == 'float':
                fea = np.concatenate([np.squeeze(pm[x]) for x in loca])
            
            fea_lab = np.concatenate([fea, np.array([float(da[i, x]) for x in [6,7,9,10,11,8]])])
            fea_lab_li.append(fea_lab)
        
        fea_lab_arr = np.stack(fea_lab_li, 0)
        all_fea_lab.append(fea_lab_arr)
    
    return all_fea_lab

# run all
def ext_run_all(folder, sel_num, fea_ty):
    cat_loca, all_loca = get_loca(folder, sel_num)
    
    return extract_fea(folder, all_loca, fea_ty)

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





    