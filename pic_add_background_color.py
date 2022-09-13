# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2022/4/21-16:17
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


work_dir = r'C:\Users\PKU\Desktop\white_bg_imgs\bg_removed'
os.chdir(work_dir)

pic_list = [f for f in glob.glob(r"*.png")]

fill_color = (170, 170, 170)
filled_folder = 'gray_170'

new_folder = r'{}'.format(filled_folder)
try:
    os.mkdir(new_folder)
except:
    print('new folder already exists! ')

for idx_pic_name in range(len(pic_list)):
    i_pic_name = pic_list[idx_pic_name]
    out_pic_name = i_pic_name[-7:-4]

    img = Image.open(i_pic_name).convert("RGBA")
    background = Image.new(img.mode[:-1], img.size, fill_color)
    background.paste(img, img.split()[-1])  # omit transparency
    img = background
    img.convert("RGB").save(fr'.\{new_folder}\{out_pic_name}.tif')
    cv2img = np.array(img)

    new_img = cv2.resize(cv2img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(fr'.\{new_folder}\{out_pic_name}.tif', new_img)

print('finished! ')
# cv2.imshow('img', new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

