
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:02:07 2018
@author: alec
Contains functions for image background subtraction & 
batch processing using chosen algorithm

requires grayscale images
# TODO: add functionality for RGB images
"""

# imports
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from pathlib import Path

def norm_pdf(x, mean, sigma):
  return(1/(np.sqrt(2*np.pi)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))


def backsub_imgs(img_dir, save_dir, method,f_increment=1,paired=False,n_frames='all'):
  " Choose a method and perform batch background subtraction "
  imgs = list()
  if n_frames == 'all':
    for file in Path(img_dir).iterdir():
        if not file.is_file():
            continue

        imgs.append(iio.imread(file))
  else:
    for file in Path(img_dir).iterdir()[0:n_frames]:
    if not file.is_file():
        continue

    imgs.append(iio.imread(file))
  
  case "min_sub":
    backsubed = min_sub(imgs,paired,f_increment)
  case "mean_sub":
    backsubed = mean_sub(imgs,paired,f_increment)
  
  
  # if save_dir doesn't exist, create it
  # imwrite in save_dir
  
def moving_min():
  
  
  
def min_sub(imgs,paired=False,f_increment=1,n_frames='all'):
  " input list of images "
  if paired==False:
    print('Continuous background subtraction')
    print('Finding minimum')
    d = imgs[0].max*np.ones(img[0].shape)
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
    d1 = imgs[0].max*np.ones(img[0].shape)
    d2 = imgs[1].max*np.ones(img[1].shape)
    for f in np.arange(0,len(imgs),2)
      d1 = np.minimum(d1,imgs[f])
      d2 = np.minimum(d2,imgs[f+1])
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for f in np.arange(0,len(imgs),2):
      backsubed.append(imgs[f] - d1)
      backsubed.append(imgs[f+1] - d2)
   
  return(backsubed)
  
def mean_sub(imgs,threshold,n_frames='all'):
  " input list of images "
  if paired==False:
    print('Continuous background subtraction')
    print('Finding minimum')
    d = np.zeros(img[0].shape)
    frames = np.arange(0,len(imgs),f_increment)
    for f in frames:
      d += f
    d = d/len(frames)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for img in imgs:
      backsubed.append(img - d)

      
  elif paired==True:
    print('Paired background subtraction')
    print('Finding minimum')
    d1 = np.zeros(img[0].shape)
    d2 = np.zeros(img[1].shape)
    for f in np.arange(0,len(imgs),2)
      d1 += imgs[f]
      d2 += imgs[f+1]
    d1 = d1/(len(imgs)/2)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for f in np.arange(0,len(imgs),2):
      backsubed.append(imgs[f] - d1)
      backsubed.append(imgs[f+1] - d2)
   
  return(backsubed)
  
  
def median_sub(imgs,n_frames='all'):
  
  
def gaussian_avg(imgs,n_frames='all'):
  
def GMM_sub(imgs, n_frames='all'):
  bsub = cv2.createBackgroundSubtractorGMG('InitializationFrames',100,'DecisionThreshold',20)
  
  
