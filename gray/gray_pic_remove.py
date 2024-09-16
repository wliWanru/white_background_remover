# python script
# -*- coding: utf-8 LF
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2023/11/16-15:09
"""
import cv2
import numpy as np
import os
import glob
from pymatting import cutout

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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert to BGRA (with alpha channel)

    # Initialize mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define background and foreground models (used by the algorithm)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object to help GrabCut to separate foreground from background
    rect = (1, 1, img.shape[1]-2, img.shape[0]-2)  # Rectangle covering the whole image

    # Run GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Set probable and definite foreground to 1 in the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Refine the mask using alpha matting
    # You might need to adjust the parameters or use a different alpha matting technique
    # For example, using the following function:
    # alpha = estimate_alpha_cf(img, mask2)

    # Use the mask to create the final image
    img = img * mask2[:, :, np.newaxis]

    # Save the result
    out_pic_name = pic_name.split('.')[0] + '_bg_removed.png'
    cv2.imwrite(os.path.join(new_folder, out_pic_name), img)
