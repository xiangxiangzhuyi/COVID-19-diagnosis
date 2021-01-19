# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:47:01 2020

@author: pc
"""


import matplotlib.pyplot as plt
import support_based as spb
import os
import numpy as np
import cv2

# set the image path
path = spb.pro_path + '/raw_data/ct_lesion_seg/'

def read_i_m():
    i_p = path + 'image/'
    m_p = path + 'mask/'
    m_d = os.listdir(m_p)
    
    # read
    i_li = []
    m_li = []
    for fo in m_d:
        fu_fo = m_p + fo
        fo_d = os.listdir(fu_fo)
        for fi in fo_d:
            m_f = fu_fo + '/' + fi
            i_f = i_p + fo + '/' + fi.split('.')[0] + '.jpg'
            im = plt.imread(i_f)
            ma = plt.imread(m_f)
            i_li.append(im[:,:, 0])
            m_li.append(ma)
    
    return np.stack(i_li), (np.stack(m_li)*255).astype(np.uint8)

def read_lung_m():
    img, mask = read_i_m()
    mask[mask>0] = 1
    return img, mask

# divide the dataset    
def div_set(img, mask, tr_rate):
    # disorder the image
    '''
    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img = img[index]
    mask = mask[index]
    '''
    
    # divide
    dn = np.ceil(img.shape[0]*tr_rate).astype(int)
    
    return img[0:dn], mask[0:dn], img[dn:], mask[dn:]   


# read image by dataset's name
def read_data(aug_str, aug_num, onl_lun, resi_str):
    if aug_str == 'no':
        d = np.load(spb.mo_path + 'select_data/noaug.npz')
    else:
        d = np.load(spb.mo_path + 'select_data/' + str(aug_num) + '_aug.npz')
    
    tr_i = d['tr_i']
    te_i = d['te_i']
    tr_m = d['tr_m']
    te_m = d['te_m']
    
    # whether only keep the lung region
    if onl_lun == 'yes':
        tr_m[tr_m > 0] = 1
        te_m[te_m > 0] = 1
    
    # whether resize the image
    if resi_str == 'yes':
        tr_i = spb.resize(tr_i, 128, 128)
        te_i = spb.resize(te_i, 128, 128)
        tr_m = spb.resize(tr_m, 128, 128)
        te_m = spb.resize(te_m, 128, 128)
    
    # check for covidlesion
    ca_n = np.max(tr_m) + 1
        
    tr_i = np.expand_dims(tr_i, -1)
    te_i = np.expand_dims(te_i, -1)
    tr_m = spb.conv_one_hot(tr_m, ca_n)
    te_m = spb.conv_one_hot(te_m, ca_n)
    return tr_i, tr_m, te_i, te_m






    
