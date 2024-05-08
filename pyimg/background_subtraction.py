
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


def backsub_imgs(imgs, masks, save_dir, filenames, method, f_increment=1, paired=False, n_frames='all', start_frame=0, width=1000):
  " Choose a method and perform batch background subtraction "
  # imgs = list()
  # if n_frames == 'all':
  #   for file in Path(img_dir).iterdir():
  #       if not file.is_file():
  #           continue

  #       imgs.append(file)
  # else:
  #   for file in Path(img_dir).iterdir()[start_frame:n_frames]:
  #     if not file.is_file():
  #       continue

  #   imgs.append(file)

  match method:
    case "min_sub":
      backsubed = min_sub(imgs, paired, f_increment)
    case "mean_sub":
      backsubed = mean_sub(imgs, paired, f_increment)
    case "moving_min":
      backsubed = moving_min(imgs, masks, save_dir, filenames,
                             paired, f_increment, width)

  # if save_dir doesn't exist, create it
  # imwrite in save_dir


def moving_min(imgs, masks,save_dir, filenames,paired=False, f_increment=1, width):
  " input list of images "
  #backsubed = list()
  if paired == False:
    print('Continuous background subtraction')
    print('Finding minimum')

    img0 = ~np.load(masks[0]).T*iio.imread(imgs[0])
    frames = np.arange(0, len(imgs), f_increment)
    for fi, f in enumerate(frames):
      d = img0.max*np.ones(img0.shape)
      if fi < width:
        for i in np.arange(0, (2*width)):
          d = np.minimum(d, ~np.load(masks[i]).T*iio.imread(imgs[i]))
        iio.imwrite(save_dir+filenames[fi], imgs[fi]-d)
        #backsubed.append(imgs[fi]-d)
      elif fi+width > len(imgs):
        d_from = len(imgs)-fi
        for i in np.arange(fi-(width+d_from), fi+d_from):
          d = np.minimum(d, ~np.load(masks[i]).T*iio.imread(imgs[i]))
        iio.imwrite(save_dir+filenames[fi], imgs[fi]-d)
      else:
        for i in np.arange(fi-width, fi+width):
          d = np.minimum(d, ~np.load(masks[i]).T*iio.imread(imgs[i]))
        iio.imwrite(save_dir+filenames[fi], imgs[fi]-d)

  elif paired == True:
    print('Paired background subtraction')
    print('Finding minimum')
    width = 100
    for fi in np.arange(0, len(imgs), 2):
      d1 = imgs[0].max*np.ones(imgs[0].shape)
      d2 = imgs[0].max*np.ones(imgs[0].shape)
      if fi < width:
        for i in np.arange(0, (2*width), 2):
          d1 = np.minimum(d1, imgs[i])
          d2 = np.minimum(d1, imgs[i+1])
        backsubed.append(imgs[fi]-d1)
        backsubed.append(imgs[fi+1]-d2)
      elif fi+width > len(imgs):
        d_from = len(imgs)-fi
        for i in np.arange(fi-(width+d_from), fi+d_from, 2):
          d1 = np.minimum(d1, imgs[i])
          d2 = np.minimum(d1, imgs[i+1])
        backsubed.append(imgs[fi]-d1)
        backsubed.append(imgs[fi+1]-d2)
      else:
        for i in np.arange(fi-width, fi+width, 2):
          d1 = np.minimum(d1, imgs[i])
          d2 = np.minimum(d1, imgs[i+1])
        backsubed.append(imgs[fi]-d1)
        backsubed.append(imgs[fi+1]-d2)

  return(backsubed)


def min_sub(imgs, paired=False, f_increment=1):
  " input list of images "
  if paired == False:
    print('Continuous background subtraction')
    print('Finding minimum')
    d = imgs[0].max*np.ones(imgs[0].shape)
    frames = np.arange(0, len(imgs), f_increment)
    for f in frames:
      d = np.minimum(d, f)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for img in imgs:
      backsubed.append(img - d)

  elif paired == True:
    print('Paired background subtraction')
    print('Finding minimum')
    d1 = imgs[0].max*np.ones(imgs[0].shape)
    d2 = imgs[1].max*np.ones(imgs[1].shape)
    for f in np.arange(0, len(imgs), 2):
      d1 = np.minimum(d1, imgs[f])
      d2 = np.minimum(d2, imgs[f+1])
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for f in np.arange(0, len(imgs), 2):
      backsubed.append(imgs[f] - d1)
      backsubed.append(imgs[f+1] - d2)

  return(backsubed)


def mean_sub(img_files, paired=False, f_increm~mask.T*iio.imread(imgs[0])ent=1):
  " input list of images "
  if paired == False:
    print('Continuous background subtraction')
    print('Finding minimum')
    d = np.zeros(iio.imread(img_files[0]).shape)
    frames = np.arange(0, len(img_files), f_increment)
    for f in frames:
      d += iio.imread(img_files[f])
    d = d/len(frames)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for i in img_files:
      img = iio.imread(i)
      backsubed.append(img - d)

  elif paired == True:
    print('Paired background subtraction')
    print('Finding minimum')
    d1 = np.zeros(iio.imread(img_files[0]).shape)
    d2 = np.zeros(iio.imread(img_files[1]).shape)
    for f in np.arange(0, len(imgs), 2):
      d1 += iio.imread(img_files[f])
      d2 += iio.imread(img_files[f+1])
    d1 = d1/(len(imgs)/2)
    d2 = d2/(len(imgs)/2)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for f in np.arange(0, len(imgs), 2):
      backsubed.append(iio.imread(img_files[f]) - d1)
      backsubed.append(iio.imread(imgs[f+1]) - d2)

  return(backsubed)


def median_sub(imgs, paired=False, f_increment=1):
    " input list of images "
  if paired==False:
    print('Continuous background subtraction')
    print('Finding minimum')
    d = []
    frames = np.arange(0,len(imgs),f_increment)
    for f in frames:
      d.append(imgs[f]) auto 
    d = np.array(d,dtype='object')
    d = np.median(d,axis=0)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for img in imgs:
      backsubed.append(img - d)

      
  elif paired==True:
    print('Paired background subtraction')
    print('Finding minimum')
    d1 = []
    d2 = []
    for f in np.arange(0,len(imgs),2):
      d1.append(imgs[f])
      d2.append(imgs[f+1])
    d1 = np.array(d1,dtype='object')
    d1 = np.median(d1,axis=0)
    d2 = np.array(d2,dtype='object')
    d2 = np.median(d2,axis=0)
    print('Performing backgroud subtraction\n')
    backsubed = list()
    for f in np.arange(0,len(imgs),2):
      backsubed.append(imgs[f] - d1)
      backsubed.append(imgs[f+1] - d2)
   
  return(backsubed)
  
def gaussian_avg(imgs,n_frames='all'):
  
def GMM_sub(imgs, n_frames='all'):
  bsub = cv2.createBackgroundSubtractorGMG('InitializationFrames',100,'DecisionThreshold',20)
  
def apply_mask(mask,img):
  return(~mask.T*iio.imread(img)
         
def main():

    path = "/share/crsp/lab/tirthab/alecjp/2023_11_blodgett/zoom/images/grayscale/p3/"
    mask_path = "/dfs9/tirthab/alecjp/2023_11_blodgett/zoom/masks/p3/"
    imgs = glob.glob(path+'*.tif')
    masks = glob.glob(mask_path+'*.npy')
    save_path = "/dfs9/tirthab/alecjp/2023_11_blodgett/zoom/bgsub/p3/"
    method = "moving_min"
    filenames = []
    f_increment = 1
    paired = False
    n_frames = 'all'
    start_frame = 0
    width = 1000
    for i in imgs:
        filenames.append(Path(os.path.basename(i)).stem)
    
    backsub_imgs(imgs,masks,save_path,filenames,method,f_increment,paired,n_frames,start_frame,width)

if __name__ == "__main__":
    main()
