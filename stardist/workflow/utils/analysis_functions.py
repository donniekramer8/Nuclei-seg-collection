import os
import numpy as np
import cv2
import json
from scipy.io import loadmat
import h5py
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
import copy


# basic morphology features

def cntarea(cnt):
    cnt = np.array(cnt).astype(np.float32)
    area = cv2.contourArea(cnt)
    return area

def cntperi(cnt):
    cnt = np.array(cnt).astype(np.float32)
    perimeter = cv2.arcLength(cnt,True)
    return perimeter

def cntMA(cnt):
    cnt = np.array(cnt).astype(np.float32)
    #Orientation, Aspect_ratio
    [(x,y),(MA,ma),orientation] = cv2.fitEllipse(cnt)
    return [np.max((MA,ma)),np.min((MA,ma)),orientation]


# functions to adjust format of contours list

def fix_contours(contours):
    contours_fixed = []
    for polygon in contours:
        coords = np.array([list(zip(x, y)) for x, y in [polygon[0]]][0], dtype=np.int32)
        contours_fixed.append(coords)
    contours_fixed = np.array(contours_fixed)
    return contours_fixed


def adjust_contours(contour, crop_x, crop_y):
    for i, xy in enumerate(contour):
        x = xy[0] - crop_x
        y = xy[1] - crop_y

        contour[i] = [x, y]
    return contour


def get_rbg_avg(centroid, contour_raw, offset, HE_20x_WSI):
    """gets RBG average intensities inside of a contour given the image and centroid
    It is fast because it crops the image so that the image size is offset*2 width/height.
    Python passes HE_20x_WSI as reference so it shouldn't affect performance passing a
    hugh variable like this."""

    x_low = centroid[0] - offset
    x_high = centroid[0] + offset
    y_low = centroid[1] - offset
    y_high = centroid[1] + offset

    img_shape = HE_20x_WSI.shape

    # if bad shape, return -1 for each intensity mean
    if offset > centroid[0] or offset > centroid[1] or centroid[0] > (img_shape[0] - offset) or centroid[1] > (
            img_shape[1] - offset):
        print(f'centroid passed: {centroid}')
        r_avg = -1
        g_avg = -1
        b_avg = -1
        return r_avg, g_avg, b_avg

    im_crop = np.array(HE_20x_WSI[x_low:x_high, y_low:y_high], dtype=np.uint16)

    # plt.imshow(im_crop)

    crop_x = centroid[0] - offset - 1
    crop_y = centroid[1] - offset - 1

    contour_adj = adjust_contours(contour_raw, crop_x, crop_y)
    contour_new = contour_adj  # .reshape((-1,1,2)).astype(np.uint16)
    rev_contour = contour_new[:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # rev_contour = contour_new[:,:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # print(rev_contour)

    # coords NEEDS to be np.int32 matrix --> 2 columns x y

    # Create a single-channel mask
    mask = np.zeros_like(im_crop[:, :, 0], dtype=np.uint16)  # make black image of same size, will fill with mask

    # Draw contours on the single-channel mask
    # cv2.drawContours(im_crop, [rev_contour], -1, (0,255,0)) #, thickness=cv2.FILLED)  # this one makes it green so that you can see contour
    cv2.drawContours(mask, [rev_contour], 0, (1), thickness=cv2.FILLED)

    # plt.imshow(im_crop)

    r_pixels = im_crop[:, :, 0] * mask  # pixels inside mask are 1, outside == 0
    g_pixels = im_crop[:, :, 1] * mask
    b_pixels = im_crop[:, :, 2] * mask

    num_pixels = np.count_nonzero(mask)

    if num_pixels != 0:

        r_avg = round(np.sum(r_pixels) / num_pixels, 2)
        g_avg = round(np.sum(g_pixels) / num_pixels, 2)
        b_avg = round(np.sum(b_pixels) / num_pixels, 2)

    else:
        print('ZERO PIXEL')
        r_avg = -1
        g_avg = -1
        b_avg = -1

    # plt.imshow(im_crop)

    return r_avg, g_avg, b_avg