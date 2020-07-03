import numpy as np

from skimage import io, transform
from SIMulator_functions import *
import glob
import os
import argparse
import matplotlib.pyplot as plt

# ------------ Parameters-------------
def GetParams():
    opt = argparse.Namespace()

    # orientation offset
    opt.ModFac = 0.3+0.2*np.random.rand()
    # orientation offset
    opt.theta = 2*pi*np.random.rand()
    # in percentage
    opt.NoiseLevel = 18 + 8*(np.random.rand()-0.5)
    # Poisson noise extent
    opt.Poisson = np.random.randint(10000,50000)
    # NA
    opt.NA = 1.2
    # Emission wavelength
    opt.emission = np.random.randint(500,680)
    # Pattern frequencies  
    kMax = (2*opt.NA)/(opt.emission)
    factor = 0.75+0.25*np.random.rand()
    opt.offset = factor
    opt.k2 = (factor-(0.1+np.random.rand()*0.3))*(kMax)
    # Pixel Size
    opt.Psize = np.random.randint(84,110)
    # Location to save test images
    opt.sloc = "D:/User/Edward/Documents/GitHub/ML-SIM/SIMfix/Data generation/"
    # Out of focus depth
    opt.depthO = 800+200*np.random.rand()
    
    return opt

# ------------ Options --------------

files = glob.glob('D:/User/Edward/Documents/GitHub/ML-SIM/SIMfix/Data generation/*.png')
files[0]

Io = io.imread(files[0]) / 255
Io = Io - 0.3*np.amax(Io)
Io[Io<0] = 0

n_rep = 1
sNum = 1
for n_it in range(n_rep):
    
    # ------------ Main loop --------------
    for file in files:
        Io = io.imread(file) 
        Io = Io / np.amax(Io)
        Io = Io - 0.3 # Cropping LUT like this increases the number of sharp details in the image 
        Io[Io<0] = 0
        Io = Io/np.amax(Io)
        
        minDim = np.amin(Io[:,:,0].shape)
        # This is a very ugly way of re-sampling regions of the image
        if (n_it % 2) ==0:
            Io = Io[0:minDim,0:minDim,:]
           
        else:
            Io = np.rot90(Io,2)
            Io = Io[0:minDim,0:minDim,:]
            
        Io = transform.resize(Io, (512, 512), anti_aliasing=True)
        
        if np.random.rand(1) > 0.65: # 35 percent of the time use the same image for background light

            # Use same image
            Oi = Io[:,:,np.random.randint(1,3)]  # if not grayscale
            Oi = transform.resize(Oi, (512, 512), anti_aliasing=True)
            Io = Io[:,:,0]

        else:
            # Use another image
            if  Io.shape[2] > 1:
                Io = Io.mean(2)

            fnew = files[np.random.randint(0, len(files))]
            Oi = io.imread(fnew) 
            Oi = Oi / np.amax(Oi)
            Oi = Oi - 0.3
            Oi[Oi<0] = 0

            minDim = np.amin(Oi[:,:,0].shape)
            Oi = Oi[0:minDim-1,0:minDim-1,:]
            Oi = transform.resize(Oi, (512, 512), anti_aliasing=True)

            if  Oi.shape[2] > 1: # if not grayscale
                Oi = Oi.mean(2)  
                Oi = Oi/np.amax(Oi)

        Io = np.rot90(Io,n_it)
        Oi = np.rot90(Oi,n_it)
        opt = GetParams()
        stack = Generate_SIM_frame(opt, Io, Oi)

        # normalise
        for i in range(len(stack)):
            stack[i] = (stack[i] - np.min(stack[i])) / \
                (np.max(stack[i]) - np.min(stack[i]))

        stack = (stack * 255).astype('uint8')

        svPath = opt.sloc+str(sNum)+ '.tif'
        io.imsave(svPath,stack)
        
        print('Done image [%d/%d]' % (sNum+1, n_rep*len(files)),end='\r')
            
        sNum += 1