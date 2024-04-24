import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import norm
from scipy.stats import binned_statistic as bs
from skimage.morphology import (
    dilation,
    disk,
    square,
    binary_erosion,
    binary_dilation,
    erosion,
)
import cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter, median_filter
import copy
import os
import glob
import multiprocessing as mp
from itertools import repeat
from scipy.signal import detrend
from scipy.signal import convolve


def find_contours(img, threshold):
    binary_img = binary_fill_holes(img > threshold)

    contours, hierarchy = cv2.findContours(
        binary_img.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    return contours

def mask_from_contours(contour_img):

    return binary_fill_holes(contour_img)


def dilate_mask(mask, kernel_size=15, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype("uint8"), kernel, iterations=iterations)
    return dilated

def check_threshold(im):
    plt.figure()
    plt.pcolormesh(im, cmap="gray")
    plt.show()
    good = 0
    while good == 0:
        img = copy.deepcopy(im)
        threshold = input("Enter intensity threshold: \n")
        validt = 0
        while validt == 0:
            try:
                t = int(threshold)
                validt = 1
            except:
                threshold = input("Enter VALID intensity threshold \n")
                validt = 0
        img[img < t] = 0
        plt.figure()
        plt.pcolormesh(img, cmap="gray")
        plt.show()
        g = input("Accept thresholding? (y/n) \n")
        to_continue = 0
        while to_continue == 0:
            if g == "y" or g == "Y":
                good = 1
                to_continue = 1
            elif g == "n" or g == "N":
                to_continue = 1
            else:
                g = input("Please enter (y/n): \n")

    plt.close("all")
    return t

def find_flame_contour(img_frame, threshold,size_threshold):

    img_bin = img_frame > threshold
    contours = find_contours(img_frame, threshold)
    c_len = np.array([len(c) for c in contours])
    """flame_contour = contours[np.where(c_len == np.max(c_len))[0][0]]
    if 
    try:
        cnt_bottom = np.where(np.squeeze(flame_contour, axis=1)[:, 1] == 1)[0]
    except:
        cnt_bottom = np.where(np.squeeze(flame_contour, axis=1)[:, 1] == 0)[0]
    fill_lims = [
        np.squeeze(flame_contour[cnt_bottom])[:, 0].min(),
        np.squeeze(flame_contour[cnt_bottom])[:, 0].max(),
    ]
    img_bin[0, fill_lims[0] : fill_lims[1]] = 1
    contours = find_contours(img_bin, 0.5)
    c_len = np.array([len(c) for c in contours])
    return contours[np.where(c_len == np.max(c_len))[0][0]]"""
    flame_contours = []
    for c in np.where(c_len > size_threshold)[0]:
        flame_contours.append(contours[c])
    return(flame_contours)


def make_mask(params):
  img_file,threshold,size_threshold = params
  img_frame = iio.imread(img_file).T
  if not threshold or threshold == 0:
        threshold = check_threshold(img_frame)
  flame_contours = find_flame_contour(img_frame, threshold,size_threshold)
  img_flame = np.zeros(img_frame.shape, np.uint8)
  for c in flame_contours:
      cnt_pts = np.squeeze(c, axis=1)
      img_flame[cnt_pts[:, 1], cnt_pts[:, 0]] = 1
  mask = mask_from_contours(img_flame)
  return(mask)

def save_masks(params):
  img_file,threshold,size_threshold,save_path,filename = params
  img_frame = iio.imread(img_file).T
  if not threshold or threshold == 0:
        threshold = check_threshold(img_frame)
  flame_contours = find_flame_contour(img_frame, threshold,size_threshold)
  img_flame = np.zeros(img_frame.shape, np.uint8)
  for c in flame_contours:
      cnt_pts = np.squeeze(c, axis=1)
      img_flame[cnt_pts[:, 1], cnt_pts[:, 0]] = 1
  mask = mask_from_contours(img_flame)
  iio.imwrite(save_path+filename, mask)
  
def main():
    threshold = 24
    size_threshold = 80
    path = "/share/crsp/lab/tirthab/alecjp/2023_11_blodgett/zoom/images/grayscale/p1/"
    imgs = glob.glob(path+'*.tif')
    save_path = "/share/crsp/lab/tirthab/alecjp/2023_11_blodgett/zoom/images/masks/p1/"
    filenames = []
    for i in imgs:
        filenames.append(os.path.basename(imgs[i]))
    cores = 10
    f_tot = len(imgs)
    objList = list(
        zip(
            imgs,
            repeat(threshold, times=f_tot),
            repeat(size_threshold, times=f_tot),
            repeat(save_path, times=f_tot),
            filenames
        )
    )

    pool = mp.Pool(processes=cores)
    pool.map(save_mask, objList)


if __name__ == "__main__":
    main()
