# python script
# -*- coding: utf-8 LF
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2023/11/16-15:05
"""
import cv2
import numpy as np
import glob
import os

# Set your work directory
work_dir = r'C:\Users\PKU\Desktop\H\fmri_0921\Img_vault\FOB2023short'
os.chdir(work_dir)

new_folder = r'mask'
try:
    os.mkdir(new_folder)
except:
    print('new folder already exists!')

pic_list = [f for f in glob.glob(r"*.png")]
resize_times = 2
gray_value = 128  # Target gray value
tolerance = 10    # Tolerance value to target shades around the gray value

for idx_pic_name in range(len(pic_list)):
    i_pic_name = pic_list[idx_pic_name]
    out_pic_name = i_pic_name[0:3]
    img = cv2.imread(i_pic_name)
    rows, cols, channels = img.shape

    # Resize the image
    img = cv2.resize(img, (cols * resize_times, rows * resize_times), interpolation=cv2.INTER_CUBIC)

    # Create a mask where the gray areas are white and everything else is black
    lower_gray = np.array([gray_value - tolerance] * 3)
    upper_gray = np.array([gray_value + tolerance] * 3)
    mask = cv2.inRange(img, lower_gray, upper_gray)

    # You might need to refine the mask using morphological operations like erode and dilate here

    # Create an alpha channel based on the inverted mask
    mask_alpha = cv2.bitwise_not(mask)

    # Split the original image into its color components
    b, g, r = cv2.split(img)

    # Merge with the new alpha channel
    bgra = cv2.merge([b, g, r, mask_alpha])

    # Save the result
    cv2.imwrite(fr'.\{new_folder}\{out_pic_name}_gray_removed.png', bgra)
