
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
  
  
  
  
def min_sub(imgs,paired=False,f_increment=1):
  " input list of images "
  if paired==False:
    print('Continuous background subtraction')
    print('Finding minimum')
    d = imgs[0].max*np.ones(img.shape())
    frames = np.arange(0,len(imgs),f_increment)
    for f in frames:
      d = np.minimum(d,f)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for img in imgs:
      backsubed.append(img - d)

      
  elif paired==True:
    print('Paired background subtraction')
    print('Finding minimum')
    d1 = imgs[0].max*np.ones(img.shape())
    d2 = imgs[1].max*np.ones(img.shape())
    for f in np.arange(0,len(imgs),2)
      d1 = np.minimum(d1,imgs[f])
      d2 = np.minimum(d2,imgs[f+1])
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for f in np.arange(0,len(imgs),2):
      backsubed.append(imgs[f] - d1)
      backsubed.append(imgs[f+1] - d2)
   
  return(backsubed)
  
def mean_sub(imgs,n_frames='all'):
  
  
def median_sub(imgs,n_frames='all'):
  
  
def gaussian_avg(imgs,n_frames='all'):
  
def GMM_sub(imgs, n_frames='all'):
  
