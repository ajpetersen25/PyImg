
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:02:07 2018
@author: alec
Contains functions for image background subtraction & 
batch processing using chosen algorithm
"""

# imports
import numpy as np
import imageio.v3 as iio
from pathlib import Path


def backsub_imgs(img_dir, method):
  " Choose a method and perform batch background subtraction "
  imgs = list()
  for file in Path(img_dir).iterdir():
      if not file.is_file():
          continue

      imgs.append(iio.imread(file))
  
  
  
  
def min_sub(img):
  
  
def mean_sub(img,n_frames='all'):
  
  
def median_sub(img,n_frames='all'):
  
  
def gaussian_avg(img,n_frames='all'):
  
def GMM_sub(img, n_frames='all'):
  
