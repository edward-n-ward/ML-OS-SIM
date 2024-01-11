import numpy as np
from tqdm.notebook import tqdm
import glob
import sys
import os
from PIL import Image
from skimage import io

import matplotlib.pyplot as plt


folder = 'E:/attenborough-9k-1024/attenborough-frames'
y=[x[0] for x in os.walk(folder)]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1) 
ax2 = fig.add_subplot(1, 2, 2)
plt.ion()





for f in range(len(y)-3856):
    files = glob.glob(y[f+3856]+'\*.jpg')
    if len(files)>8:

        read1 = io.imread(files[0]) 
        read2 = io.imread(files[8]) 
        ax1.clear()
        ax2.clear()
        ax1.imshow(read1)
        ax2.imshow(read2)
        fig.canvas.draw()
        plt.show()
        