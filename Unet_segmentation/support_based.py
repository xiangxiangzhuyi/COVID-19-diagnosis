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
import scipy as sp
import tifffile as tiff
import skimage.measure as skm

# set path
if os.path.exists('E:/Project15_COVID19_CT/') :
    pro_path = 'E:/Project15_COVID19_CT/'
else:
    pro_path = '/project/huangli/Project/Project15_COVID19_CT/'
mo_path = pro_path + 'Models/model04_seg_all_file/'

import sys
sys.path.insert(1, pro_path + 'Models/utiles/')
import read_all_img as rai


# combine
def com_index(sam_size, sam_num):
    ind_list  = np.arange(sam_size, dtype=np.uint16)
    str_code = 'np.array(np.meshgrid('
    for j in range(sam_num):
        str_code = str_code + 'ind_list,'
    str_code = str_code + ')).T.reshape(-1, sam_num)'
    index = eval(str_code)
    
    so_ind = np.arange(index.shape[0])
    np.random.shuffle(so_ind)
    
    return index[so_ind]

# read training images
def read_tr_img(img, mask, bat_index, w_num, h_num):
    w_p = img.shape[2]/w_num
    h_p = img.shape[1]/h_num    
    # splice images
    img_li = []
    for i in range(bat_index.shape[0]):
        # hist match
        all_im = [img[x,:,:,:] for x in bat_index[i, :]]
        all_ma = [mask[x,:,:,:] for x in bat_index[i, :]]
        mean_im = np.mean(np.stack(all_im, 0), 0).astype(np.uint16)
        all_im = [hist_match(x, mean_im) for x in all_im]
        all_im_ma = [np.concatenate([all_im[x], all_ma[x]], -1) for x in range(len(all_im))]
        # splicing
        w_li = []
        for w in range(w_num):
            h_li = []
            for h in range(h_num):
                ind = w*h_num + h
                w1 = np.ceil(w * w_p).astype(np.uint16)
                w2 = np.ceil((w + 1) * w_p).astype(np.uint16)
                h1 = np.ceil(h * h_p).astype(np.uint16)
                h2 = np.ceil((h + 1) * h_p).astype(np.uint16)
                im_p = all_im_ma[ind][h1:h2, w1:w2, :]
                h_li.append(im_p)
            h_img = np.concatenate(h_li, 0)
            w_li.append(h_img)
        w_img= np.concatenate(w_li, 1)
        img_li.append(w_img)
    
    img_arr = np.stack(img_li, 0).astype(np.uint16)
    return img_arr[:,:,:, 0:img.shape[-1]], img_arr[:,:,:, img.shape[-1]:]
    

def hist_match2(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    return interp_t_values[bin_idx].reshape(oldshape)


def hist_match(original, specified):
 
    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()
 
    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)
 
    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
 
    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')
 
    return b[bin_idx].reshape(oldshape)

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

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

# calculate global loss -------------------------------------------------
def ch_glob(tr_loss):
    tr_loss = np.stack(tr_loss, 0)
    zero_ind = np.where(tr_loss[:, 1] == 0)[0]
    if len(zero_ind) == 0:
        return False
    
    # calculate global loss
    zero_ind = np.concatenate([np.array([0]), zero_ind])
    gl_loss = []
    for i in range(len(zero_ind)- 1):
        los = tr_loss[zero_ind[i]:zero_ind[i+1], 2] * tr_loss[zero_ind[i]:zero_ind[i+1], 3]
        gl_loss.append(np.sum(los))
    
    # check
    if len(gl_loss) >= 2:
        if tr_loss.shape[0] > 1000 and gl_loss[-1] >= gl_loss[-2]:
            return True
        else:
            return False
 
def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''
    
    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1: ]    

# prediction to mask
def pred_to_mask(pred_arr):
    max_ind = np.argmax(pred_arr, axis = -1)
    
    bb = []
    for i in range(pred_arr.shape[-1]):
        block = np.zeros(pred_arr.shape[0:-1])
        block[max_ind == i] = 1
        bb.append(block)
    cc = np.stack(bb, -1)
    
    return cc

# dice 1
def dice1(mask, pred):
    a = np.sum(mask*pred)*2
    b = np.sum(mask + pred)
    return a/b

# dice 2
def dice2(mask, pred):
    a = np.sum(mask*pred, axis = (1, 2, 3))*2
    b = np.sum(mask + pred, axis = (1, 2, 3))
    di_mat = a/b
    return np.mean(di_mat)

# dice 3
def dice3(mask, pred):
    di_mat = np.sum(mask*pred, axis = (0, 1, 2))*2/np.sum(mask + pred, axis = (0, 1, 2))
    return np.mean(di_mat)


def mean_dice(mask, pred):
    di_mat = np.sum(mask*pred, axis = (1, 2))*2/(np.sum(mask + pred, axis = (1, 2)) + 1e-10)
    s_n = np.sum(mask, axis = (1, 2))
    s_n[s_n> 0] = 1
    sam_dice = np.sum(di_mat, axis = 1)/np.sum(s_n, axis = 1)
    cat_dice = np.sum(di_mat, axis = 0)/np.sum(s_n, axis = 0)
    all_dice = np.mean(cat_dice)
    
    return di_mat, sam_dice, cat_dice, all_dice

# save the model result
def save_result(result, strs, fi):
    # create folder
    folder_name = mo_path + 'save_results/' + strs + '/'
    if  not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # file name list
    np.save(folder_name + fi, result)   
        
    return

# save as png images
def save_images(img, strs, fi):
    folder_name = mo_path + 'save_results/' + strs
    if  not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        
    if img.dtype == 'float32':
        img = (img*255).astype(np.uint8)
    
    if img.shape[-1] < 128:
        img = np.argmax(img, -1).astype(np.uint8)
        img = img*50
    
    cv2.imwrite(folder_name + fi, img)
    return

# save the batch of images
def save_bat_imgs(img, mask, pred, sa_str, resi_str):
    if resi_str == 'yes':
        img = resize(img, 512, 512)        
        mask = resize(mask, 512, 512)
        pred = resize(pred, 512, 512)
    else:
        img = img[:,:,:,0]
        
    for j in range(img.shape[0]):
        save_images(img[j], sa_str + '/te_pred/', str(j) + '_img.png')
        save_images(mask[j], sa_str + '/te_pred/', str(j) + '_mask.png')
        save_images(pred[j], sa_str + '/te_pred/', str(j) + '_pred.png')
    
    return


# resize all image
def resize(al_img, w, h):
    img_li = []
    for i in range(al_img.shape[0]):
        img = cv2.resize(al_img[i], (w, h), interpolation = cv2.INTER_AREA)
        img_li.append(img)
        
    return np.stack(img_li, 0)

# calculate the original DICE
def cal_or_dice(mask, pred, resi_str):
    if resi_str == 'yes':
        pred = resize(pred, 512, 512)
    pred_mask = pred_to_mask(pred)
    Dice = mean_dice(mask, pred_mask)
    
    return Dice
    
# process mask
def pro_mask(mask):
    # closing and opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # connected component analysis
    labeled_img, num = skm.label(mask, neighbors=8, background=0, return_num=True)
    
    reg_li = skm.regionprops(labeled_img)
    for reg in reg_li:
        cent = reg.centroid
        lab_num = reg.label
        thr1 = mask.shape[0]/4
        thr2 = 3*mask.shape[0]/4
        if min(cent) < thr1 or max(cent) > thr2:
            labeled_img[labeled_img == lab_num] = 0
    labeled_img[labeled_img > 0] = 1

    return labeled_img.astype(np.uint8)

# process the batch of images
def pro_bat_mask(bat_mask):
    bat_li = [bat_mask[x] for x in range(bat_mask.shape[0])]

    return np.stack(bat_li, 0)

# debug function
def print_var(num, varb):
    print('num:' + str(num), 'max:' + str(np.max(varb)), 
          'min:' + str(np.min(varb)), 'shape:' + str(varb.shape), 
          'dtype:' + str(varb.dtype))
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

# read the prediction result
def read_pred(folder_name):
    folder_fullname = mo_path + 'save_results/' + folder_name + '/te_pred/'
    img_li = []
    mask_li = []
    pred_li = []
    for i in range(2000):
        # file full names
        img_name = folder_fullname + str(i) + '_img.png'
        mask_name = folder_fullname + str(i) + '_mask.png'
        pred_name = folder_fullname + str(i) + '_pred.png'
        if not os.path.isfile(img_name):
            break
        # read
        img_li.append(np.load(img_name))
        mask_li.append(np.load(mask_name))
        pred_li.append(np.load(pred_name))
    
    return np.stack(img_li, 0), np.stack(mask_li, 0), np.stack(pred_li, 0)

