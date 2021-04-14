# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:34:09 2021

@author: ginof
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rotate

from time import sleep
import sys

IMG_PATH = './data/example_image/'
SEG_PATH = './data/example_segmentation/'

TRUTH = './data/ISIC-2017_Training_Part3_GroundTruth.csv'
FEATURES = './features/features.csv'

"""FUNCTIONS FOR MASKED IMAGES"""

def get_boundaries(image):
    """Function to locate the boundaries of the lesion over the whole image.
    Takes a segmentation mask image as argument and returns the upper, lower, left and right boundaries."""

    mask = np.where(image == 1)
    left = min(mask[1])
    right = max(mask[1])
    upper = min(mask[0])
    lower = max(mask[0])
    return upper, lower, left, right

#def get_center(image): # NOT NEEDED ANYMORE ?

#    up, dw, lt, rt = get_boundaries(image)
#    center = ((up+dw)/2, (lt+rt)/2)
#    return center
    
def zoom(image):
    """Function to zoom-in (crop) the lesion from blank space. Takes a segmentation mask image as input,
    and returns the rectangle where the lesion is found."""

    up, dw, lt, rt = get_boundaries(image)
    rectangle = image[up:dw+1, lt:rt+1]
    return rectangle

def cuts(image):
    """Function to perform a double cut (vertical and horizontal) of the lesion. Takes a segmentation mask image as input,
    and returns the vertical and horizontal cuts (2 for each dimension). It handles uneven shapes."""

    center_h = image.shape[0] // 2 # The image shape contains a tuple with height and width (in pixels)
    if image.shape[0] % 2 == 0: # If the height is an even number of pixels, the cut returns 2 equal sides
        upside = image[:center_h,:]
        downside = image[center_h:,:]
    else: # If the height is an uneven number of pixels, the cut has to "share" the center, to return 2 equal sides
        upside = image[:center_h,:]
        downside = image[center_h+1:,:]
        
    center_w = image.shape[1] // 2    
    if image.shape[1] % 2 == 0:
        leftside = image[:,:center_w]
        rightside = image[:,center_w:]
    else:
        leftside = image[:,:center_w]
        rightside = image[:,center_w+1:]
 
    return upside, downside, leftside, rightside


def test_symmetry(image, rot_deg=30):
    """Function to test the symmetry of an image. Takes a segmentation mask image and the rotation degree interval and
    returns a symmetry score between zero (non-symmetric) to one (completely symmetric)."""

    assert (rot_deg <= 90) and (rot_deg >= 0), "Rotation degree should be positive and at most 90 deg"
    optimal = 0
    
    for deg in range(0,91, rot_deg):
        rot_image = skimage.transform.rotate(image, deg)
        z = zoom(rot_image)
        
        upside, downside, leftside, rightside = cuts(z)

        up_dw = np.sum(np.bitwise_and(upside.astype(int), np.flipud(downside).astype(int))) /\
        np.sum(np.bitwise_or(upside.astype(int), np.flipud(downside).astype(int)))

        lt_rt = np.sum(np.bitwise_and(leftside.astype(int), np.fliplr(rightside).astype(int))) /\
        np.sum(np.bitwise_or(leftside.astype(int), np.fliplr(rightside).astype(int)))
    
        symmetry = (up_dw+lt_rt)/2
        
        if symmetry > optimal: optimal = symmetry

    return symmetry
    
def rgb2gray(rgb):
    """Function to convert a RGB image to grayscale."""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def crop(image, mask):
    img = image.copy()
    img[mask==0] = 0
    return img

def color_std(image):
    try:
        R = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,0]
        G = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,1]
        B = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,2]
        color_std = (np.std(R) + np.std(G) + np.std(B)) /3
    except:
        color_std = 'NA'
    return color_std

def check_border(image, border=0.01, tolerance=0.2, warning=True):
    """Function to check if the lesion might be exceeding the image. Take the following arguments:
    - image: segmentation mask image to check.
    - border: the percentage of pixels to consider as a border. 10% by default.
    - tolerance: the percentage of tolerance for a lesion to be at the border of the image. 20% by default.
    - warning: boolean to indicate if a textual warning should be issue when checking the border. True by default."""
    h = int(image.shape[0] * border)
    w = int(image.shape[1] * border)
    up = (np.sum(image[h,:]) / image.shape[1]) > tolerance
    dw = (np.sum(image[-h,:]) / image.shape[1]) > tolerance
    lt = (np.sum(image[:,w]) / image.shape[0]) > tolerance
    rt = (np.sum(image[:,w]) / image.shape[0]) > tolerance
    if warning:
        if up or dw or lt or rt: return "This lesion might be overflowing the image"
        else: return "This lesion does not seem to be overflowing the image"
    else:
        return up or dw or lt or rt
    
def dimensions(mask1): # SANNA'S WAY
    '''calculates height(max) and width(90 deg to height)
        returns height, width, rotated mask image, degree of rotation'''
    pixels_in_col = np.max(np.sum(mask1, axis=0))

    rot = 0
    max_col = 0
    rot_max = 0
    for _ in range(9):
        rot_im = transform.rotate(mask1,rot)
        pixels_in_col = np.max(np.sum(rot_im, axis=0))
        if pixels_in_col > max_col:
            max_col = pixels_in_col
            rot_max = rot
            pixels_in_row = np.max(np.sum(rot_im, axis=1))
        rot += 10

    return max_col, pixels_in_row, rot_max

def laydown(image):
    z = zoom(image)
    u, d, l, r = get_boundaries(z)
    if (d-u) >= (r-l):
        return skimage.transform.rotate(image, 90) 
    else: return image
        
#def border_split(image):
    
df = pd.read_csv(TRUTH, index_col='image_id')
df = df.astype(int) # Transform to int

symmetry = []
i = 1
for ix, row in df.iterrows():
    file_path = SEG_PATH + str(ix) + "_segmentation.png"
    image = plt.imread(file_path)
    symmetry.append(test_symmetry(image))
    per = str(round((i/2000)*100))+'%'
    print('\r'+per)
    i += 1
    
df['symmetry'] = symmetry

print(df)
