# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:06:59 2020

@author: pc
"""

import model
import support_dataset as spda
import numpy as np
import support_based as spb
import time

def run_model(w_num, h_num, tr_bat, te_bat, iter_num, test_once, GPU_str, aug_str, aug_num, onl_lun, resi_str):
    # saving folder
    sa_str = spb.com_mul_str([aug_str, onl_lun, resi_str, aug_num, w_num, h_num])
    # construct dataset
    da = spda.dataset(w_num, h_num, tr_bat, te_bat, aug_str, aug_num, onl_lun, resi_str)
        
    # construct UNET
    Unet = model.UNET(da.i_w, da.i_h, da.i_ch, da.m_w, da.m_h, da.o_ch, 0.0001, sa_str, 'yes', GPU_str)
    Unet.build_framework()
    
    # create lists to save the results
    spb.os.mkdir(spb.mo_path + 'save_results/' + sa_str)
    tr_res = []
    te_res = []
    for i in range(iter_num):
        # train
        tr_img, tr_mask = da.get_tr_bat_img()
        tr_loss, tr_out = Unet.train(tr_img, tr_mask)
        tr_res.append(np.array([i, da.tr_ind, np.mean(tr_loss), tr_img.shape[0]]))
        
        print('train', i, np.mean(tr_loss), time.ctime())
        
        # test
        if (i+1)%test_once == 0:
            te_loss, te_dice, te_pred = Unet.test(da, tr_bat)
            or_dice = spb.cal_or_dice(da.or_te_m, te_pred, resi_str)
            te_res.append([i, np.mean(te_loss), te_dice, or_dice])
            
            # print
            print('test', i, np.mean(te_loss), te_dice[-1], or_dice[-1], time.ctime())
            
            # save result
            spb.save_result(np.stack(tr_res, 0), sa_str, 'tr_res.npy')
            spb.save_list(te_res, sa_str, 'te_res.npy')
            spb.save_bat_imgs(da.te_i, da.te_m, te_pred, sa_str, resi_str)
            
            if or_dice[-1] > 0.9935:
                break
    
    # segmentation all images
    al_da = spb.rai.all_imgs(10)
    for i in range(10000):
        bat_img, name_li, bat_fina = al_da.read_by_type('img')
        bat_lung_area = []
        for b_img in bat_img:
            
            b_img = np.transpose(b_img, (2, 0, 1))
            b_img = np.expand_dims(b_img, -1)
            b_lung_mask = Unet.pred(b_img)
            b_lung_mask = np.argmax(b_lung_mask, -1).astype(np.uint8)
            
            b_lung_mask = spb.pro_bat_mask(b_lung_mask)
            
            b_lung_area = b_img[:,:,:, 0]*b_lung_mask
            b_lung_area = np.transpose(b_lung_area, (1, 2, 0))
            
            bat_lung_area.append(b_lung_area)
        al_da.write_bat_img(bat_lung_area, name_li, bat_fina, 'lung_area')
        [print(x) for x in name_li]
        if al_da.img_ind == 0:
            break
    

    
            
        
        






