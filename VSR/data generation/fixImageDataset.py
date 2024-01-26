import numpy as np
from tqdm.notebook import tqdm
import glob
import sys
import os
from PIL import Image
from skimage import io

files = glob.glob('E:/datasets/validation/*.tif')
files[0:5]
img = io.imread(files[0])
img.shape
np.amax(img[4,:,:])
nCh_in = 3

# ------------ Main loop --------------
counter = 0     
for f in range(len(files)):

  file_path=files[f]
  
  try:
    img = np.array(io.imread(file_path))
    if np.amax(img[nCh_in+1,:,:]) == 0:
      os.remove(file_path)
      counter +=1
      print('Image id: '+ files[f] + ' was removed due to bad GT. Total images found: ' + str(counter))
    elif np.amax(img[0,:,:]) == 0:  
      os.remove(file_path)
      counter +=1
      print('Image id: '+ files[f] + ' was removed due to bad input. Total images found: ' + str(counter))     
  
  except:
    try:
      os.remove(file_path)
      print('Image id: '+ files[f] + ' was removed due to failed file read. Total images found: ' + str(counter)) 
      counter +=1
    except:
      print('Image id: '+ files[f] + ' failed to delete. Check file is present') 
      counter +=1  

print('Number of images removed :' + str(counter))