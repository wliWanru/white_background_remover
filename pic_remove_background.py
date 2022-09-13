# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2022/4/23-9:58
"""
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pymatting import cutout


def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


work_dir = r'C:\Users\PKU\Desktop\white_bg_imgs'
os.chdir(work_dir)

new_folder = r'bg_removed'
try:
    os.mkdir(new_folder)
except:
    print('new folder already exists! ')

pic_list = [f for f in glob.glob(r"resized_*.png")]

resize_times = 5
thresh = 254

for idx_pic_name in range(278, len(pic_list)):
    i_pic_name = pic_list[idx_pic_name]
    input_img_name = i_pic_name
    out_pic_name = i_pic_name[-7:-4]
    input_mask_name = fr'.\mask\{out_pic_name}_thresh{thresh}.png'
    output_img_name = fr'.\{new_folder}\{out_pic_name}.png'
    cutout(
       # input image path
       input_img_name,
       # input trimap path
       input_mask_name,
       # output cutout path
       output_img_name)




