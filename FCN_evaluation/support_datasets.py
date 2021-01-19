# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:01:13 2020

@author: DELL
"""

import support_based as spb
import numpy as np
import pickle
import os

# read file framework
class dataset:
    def __init__(self, bat_num, di_ind,st_str):
        with open(spb.mo_path + 'stat_res/divid_da', 'rb') as file:
            self.d_li = pickle.load(file)
        
        # assign parts into training, validation, and test
        self.da_inf = [np.concatenate([self.d_li[y] for y in x]) for x in di_ind]
        
        # only keep available scans
        self.da_inf = [x[np.where(x[:, 4].astype(np.float32) != -1000.0)] for x in self.da_inf]
        
        self.inf_li, self.ind_li, self.ind_fla = {}, {}, {}
        da_str = ['tr', 'va', 'te']
        for i in range(3):
            self.inf_li[da_str[i]] = self.da_inf[i]
            self.ind_li[da_str[i]] = np.arange(self.da_inf[i].shape[0], dtype=np.uint16)
            self.ind_fla[da_str[i]] = 0
            
        self.all_im = spb.raig.all_imgs('resize_lungarea_file_framework', 10)
        self.bat_num = bat_num
        
        # read np data
        self.st_str = st_str
        self.np_pa = spb.pro_path + 'np_data/'
        if self.st_str == 'yes':
            self.fi_li = os.listdir(self.np_pa)
            self.np_li = [np.load(self.np_pa + x ) for x in self.fi_li]
            
    # get index list   
    def get_bat_index(self, index_list, bat_num, ind):
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
    
    # randomly crop the volume
    def rand_crop(self, img):
        sub_w = 47
        h_w = int((sub_w - 1)/2)
        ind = [np.random.randint(h_w, x-h_w) for x in img.shape]

        return img[ind[0]-h_w: ind[0]+h_w+1, ind[1]-h_w: ind[1]+h_w+1, ind[2]-h_w: ind[2]+h_w+1]
    
        
    # read np data
    def read_np(self, na):
        if self.st_str == 'yes':
            ind = self.fi_li.index(na + '.npy')
            return self.np_li[ind]
        else:
            return np.load(self.np_pa + na + '.npy')
    
     # get index list   
    def get_img(self, da_str, crop_str):
        bat_index, self.ind_fla[da_str] = self.get_bat_index(self.ind_li[da_str], self.bat_num, self.ind_fla[da_str])
        inf = self.inf_li[da_str][bat_index, :]
        
        # read
        img_li, lab_li, na_li = [], [], []
        for i in range(len(bat_index)):
            img_li.append(self.read_np(inf[i, 0]))
            lab_li.append(float(inf[i, 8]))
            na_li.append(inf[i, 0])
        
        if crop_str == 'yes':
            img_li = [self.rand_crop(x) for x in img_li]
        
        return np.expand_dims(np.stack(img_li, 0), -1), np.array(lab_li).astype(np.uint8), na_li
        

        













        
        
        