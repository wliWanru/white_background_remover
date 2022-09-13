# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2022/4/23-10:26
"""
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# this should be the folder containing images to be processed
work_dir = r'C:\Users\PKU\Desktop\white_bg_imgs'
os.chdir(work_dir)

new_folder = r'mask'
try:
    os.mkdir(new_folder)
except:
    print('new folder already exists! ')

pic_list = [f for f in glob.glob(r"*.tif")]

resize_times = 5

for idx_pic_name in range(496, len(pic_list)):
    i_pic_name = pic_list[idx_pic_name]

    out_pic_name = i_pic_name[0:3]

    img = cv2.imread(i_pic_name)

    rows, cols, channels = img.shape
    # 5 times large
    img_copy = np.copy(img)

    img = cv2.resize(img, (cols * resize_times, rows * resize_times), interpolation=cv2.INTER_CUBIC)

    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = 254
    ret, thresh_img = cv2.threshold(img_cvt, thresh, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_not(thresh_img)
    mask_alpha = mask.copy()

    n_iters = 5
    kernel_dilated = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel=kernel_dilated, iterations=n_iters)
    kernel_eroded = np.ones((10, 10), np.uint8)
    eroded = cv2.erode(mask, kernel=kernel_eroded, iterations=n_iters)


    kernel_morph = np.ones((20, 20), np.uint8)

    closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_morph)
    opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_morph)

    res = dilated.copy()
    res[((dilated == 255) & (closing == 0))] = 128


    mask = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    if mask.max() < 255:
        print(i_pic_name+' max < 255! ')
        break
    b, g, r = cv2.split(mask)

    bgra = cv2.merge([b, g, r, mask_alpha])

    # bgra = cv2.resize(bgra, (cols, rows), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(fr'.\{new_folder}\{out_pic_name}_thresh{thresh}.png', bgra)

    cv2.imwrite(f'resized_{out_pic_name}.png', img)
