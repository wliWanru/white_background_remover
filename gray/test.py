# python script
# -*- coding: utf-8 LF
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2023/11/16-15:13
"""
import cv2
import numpy as np
import os
import glob
from pymatting import cutout, estimate_alpha_cf, estimate_foreground_ml

# Define the directory containing the images to be processed
work_dir = r'C:\Users\PKU\Desktop\H\fmri_0921\Img_vault\FOB2023short'
os.chdir(work_dir)

# Check and create a new directory for processed images
new_folder = r'bg_removed'
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

pic_list = glob.glob("*.png")

for pic_name in pic_list:
    # Read the image
    img = cv2.imread(pic_name)
    img_float = img.astype(np.float32) / 255.0  # Convert to float
    
    # Assume the gray value of 128 is for the background, create a trimap
    # Trimap value: 0 for background, 128 for unsure, 255 for foreground
    trimap = np.full(img.shape[:2], 128, dtype=np.uint8)
    trimap[img.mean(axis=2) < (128.0 / 255.0)] = 0  # Background
    trimap[img.mean(axis=2) > (128.0 / 255.0)] = 255  # Foreground
    
    # Estimate alpha and foreground
    alpha = estimate_alpha_cf(img_float, trimap)
    foreground = estimate_foreground_ml(img_float, alpha)
    
    # Combine the estimated foreground with the alpha matte
    foreground[alpha < 0.5] = 0
    result = np.uint8(foreground * 255)
    
    # Save the result
    out_pic_name = pic_name.split('.')[0] + '_bg_removed.png'
    cv2.imwrite(os.path.join(new_folder, out_pic_name), result)
