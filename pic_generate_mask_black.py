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


def filter_contours(mask, min_area=5000):
    """
    Filters out smaller contours and keeps only those that touch the border of the image.
    This helps in focusing on the outer background while excluding internal features.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    border_contours_mask = np.zeros_like(mask)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Check if the contour touches the image border
            touching_border = False
            for point in contour:
                x, y = point[0]
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    touching_border = True
                    break
            if touching_border:
                cv2.drawContours(border_contours_mask, [contour], -1, (255), thickness=cv2.FILLED)
    
    return border_contours_mask


#
# # this should be the folder containing images to be processed
# work_dir = r'Q:\data\project_data\beh_scene\select_blur_face\monkey'
# os.chdir(work_dir)
#
# new_folder = r'mask'
# try:
#     os.mkdir(new_folder)
# except:
#     print('new folder already exists! ')
#
# pic_list = [f for f in glob.glob(r"*.jpg")]
#
# resize_times = 2
#
# for idx_pic_name in range(0, len(pic_list)):
#     i_pic_name = pic_list[idx_pic_name]
#
#     out_pic_name = i_pic_name[0:3]
#
#     img = cv2.imread(i_pic_name)
#
#     rows, cols, channels = img.shape
#     # 5 times large
#     img_copy = np.copy(img)
#
#     img = cv2.resize(img, (cols * resize_times, rows * resize_times), interpolation=cv2.INTER_CUBIC)
#
#     img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     thresh = 10
#     ret, thresh_img = cv2.threshold(img_cvt, thresh, 255, cv2.THRESH_BINARY)
#
#     # For black background, use the threshold image directly without inversion
#     mask = thresh_img
#     mask_alpha = mask.copy()
#
#     n_iters = 5
#     kernel_dilated = np.ones((3, 3), np.uint8)
#     dilated = cv2.dilate(mask, kernel=kernel_dilated, iterations=n_iters)
#     kernel_eroded = np.ones((10, 10), np.uint8)
#     eroded = cv2.erode(mask, kernel=kernel_eroded, iterations=n_iters)
#
#     kernel_morph = np.ones((20, 20), np.uint8)
#
#     closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_morph)
#     opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_morph)
#
#     res = dilated.copy()
#     res[((dilated == 255) & (closing == 0))] = 128
#
#     mask = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
#     if mask.max() < 255:
#         print(i_pic_name + ' max < 255! ')
#         break
#     b, g, r = cv2.split(mask)
#
#     bgra = cv2.merge([b, g, r, mask_alpha])
#
#     # bgra = cv2.resize(bgra, (cols, rows), interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(fr'.\{new_folder}\{out_pic_name}_thresh{thresh}.png', bgra)
#
#     cv2.imwrite(f'resized_{out_pic_name}.png', img)
#

# Define the working directory
work_dir = r'Q:\data\project_data\beh_scene\select_blur_face\monkey'
os.chdir(work_dir)

new_folder = r'mask'
try:
    os.mkdir(new_folder)
except FileExistsError:
    print('New folder already exists!')

pic_list = [f for f in glob.glob(r"*.jpg")]

resize_times = 5

for idx_pic_name in range(0, len(pic_list)):
    i_pic_name = pic_list[idx_pic_name]
    
    out_pic_name = i_pic_name[0:3]
    
    img = cv2.imread(i_pic_name)
    
    rows, cols, channels = img.shape
    # Resize the image to make it 2x larger
    img_resized = cv2.resize(img, (cols * resize_times, rows * resize_times), interpolation=cv2.INTER_CUBIC)
    
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to detect the black background
    thresh = 5
    ret, thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    
    # Filter out small regions (like pupils) by focusing only on larger contours
    filtered_mask = filter_contours(thresh_img, min_area=5000)  # Adjust the min_area as needed
    
    # Create the alpha mask
    mask_alpha = filtered_mask.copy()
    
    # Perform dilation and erosion to refine the mask
    n_iters = 15
    kernel_dilated = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(filtered_mask, kernel_dilated, iterations=n_iters)
    kernel_eroded = np.ones((10, 10), np.uint8)
    eroded = cv2.erode(filtered_mask, kernel_eroded, iterations=n_iters)
    
    # Apply morphological operations to clean up the mask
    kernel_morph = np.ones((20, 20), np.uint8)
    closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_morph)
    opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_morph)
    
    # Combine the masks
    res = dilated.copy()
    res[((dilated == 255) & (closing == 0))] = 128
    
    # Convert mask to RGB
    mask_rgb = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    
    # Check for valid mask
    if mask_rgb.max() < 255:
        print(i_pic_name + ' max < 255! ')
        break
    
    # Split the mask into channels and create the final BGRA image
    b, g, r = cv2.split(mask_rgb)
    bgra = cv2.merge([b, g, r, mask_alpha])
    
    # Save the tri-map mask and the resized image
    cv2.imwrite(fr'.\{new_folder}\{out_pic_name}_thresh{thresh}.png', bgra)
    cv2.imwrite(f'resized_{out_pic_name}.png', img_resized)
    
    