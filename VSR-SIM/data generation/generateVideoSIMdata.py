import math
import numpy as np
import glob
from skimage import io, transform
import argparse
import os
from VideoSIMulator_functions import *

# ------------ normalisation-------------
def threshold_and_norm(arr,power):

    arr = arr-np.amin(arr)
    arr = (arr/np.amax(arr))
    hist, bins = np.histogram(arr,16,[0, 1])
    ind = np.where(hist==np.amax(hist))
    mini = bins[ind[0][0]]
    maxi = bins[ind[0][0]+1]
    sub = (maxi+mini)/2
    arr = arr - sub
    arr[arr<0]=0
    arr = (arr/np.amax(arr))
    arr = np.power(arr,power) # Increase dynamic range of sample 
    return arr

# ------------ Parameters-------------
def GetParams():
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 3
    # number of orientations of stripes
    opt.Nangles = 1
    # modulation factor
    opt.ModFac = 0.5+0.5*np.random.rand()
    # orientation offset
    opt.alpha = 0
    # orientation error
    opt.angleError = (0.5-np.random.rand())*2*np.pi
    # shuffle the order of orientations
    opt.shuffleOrientations = False
    # random phase shift errors
    opt.phaseError = ((2*np.pi)/opt.Nshifts)*(0.5-np.random.rand(opt.Nangles, opt.Nshifts))
    # in percentage
    opt.NoiseLevel = 8 + 6*(np.random.rand()-0.5)
    # Poisson noise extent
    # include OTF and GT in stack
    opt.Poisson = np.random.randint(100,1000)
    opt.OTF_and_GT = True
    # NA
    opt.NA = 1.2
    # Emission wavelength
    opt.emission = np.random.randint(500,690)
    # Pattern frequencies  
    opt.excitation = opt.emission-30
    kMax = (2*opt.NA)/(opt.emission)
    factor = 1 # Adjust for effects of aberrations
    opt.offset = factor
    opt.k2 = (0.3+0.15*np.random.rand())*(2*opt.NA)/(opt.excitation)
    # Pixel Size
    opt.Psize = np.random.randint(70,90)
    # Image to use for ground truth in training
    opt.target = 'original' # 'original' or 'SIM' 
    # Out of focus depth
    opt.depthO = 700+800*np.random.rand()
    # Data size
    opt.im_size = 512 # Model data size
    opt.scale = 1 # Ground truth upscale
    opt.pad = False
    
    # Adjusted pixel Size
    opt.Psize = opt.Psize/opt.scale
    
    return opt


folder = 'E:/attenborough-9k-1024/attenborough-frames-sorted'
staticFiles = glob.glob('E:/DIV2K/*.png')
sLoc = 'E:/datasets/OS-SIM'

y=[x[0] for x in os.walk(folder)]

n_rep = 2
sNum = 2700

print('Number of videos: ' + str(len(y)))
print('Number of repetitions: ' + str(n_rep))

print(folder)
print(staticFiles)
print(sLoc)

for n_it in range(n_rep):
    n_it = n_it+1
   
    # ------------ Main loop --------------
    for f in range(len(y)):

        files = glob.glob(y[f]+'/*.jpg')
        
        if len(files)>8:
            opt = GetParams()
            
            type = np.random.rand()
            Io = np.zeros((opt.im_size*opt.scale, opt.im_size*opt.scale,opt.Nangles*opt.Nshifts))
            rot = np.random.randint(0,3)
            power = np.random.randint(1,4)
            for p in range(3):
                
                # In focus image
                file = files[0]     
                read = io.imread(file).astype('float') 
                if  read.shape[2] > 1:
                    read = read.mean(2)
                
                minDim = np.amin(read.shape)

                # This is a very ugly way of re-sampling regions of the image
                read = np.rot90(read,rot)
                read = read[0:minDim,0:minDim]

                read = transform.resize(read, (opt.im_size*opt.scale, opt.im_size*opt.scale), anti_aliasing=True)
                read = threshold_and_norm(read,power)
                Io[:,:,p] = read
                
                
            # Out of focus image
            
            fnew = staticFiles[np.random.randint(0, len(staticFiles))]
            Oi = io.imread(fnew).astype('float')
            if  Oi.shape[2] > 1: # if not grayscale
                Oi = Oi.mean(2)  
                Oi = Oi/np.amax(Oi)

            minDim = np.amin(Oi.shape)
            Oi = np.rot90(Oi,np.random.randint(0,3))
            Oi = Oi[0:minDim-1,0:minDim-1]
            Oi = transform.resize(Oi, (opt.im_size*opt.scale, opt.im_size*opt.scale), anti_aliasing=True)
            Oi = threshold_and_norm(Oi,power)

            stack = Generate_SIM_Image(opt, Io, Oi)


            # normalise
            int_range = 20*opt.Poisson
            for i in range(len(stack)):
                stack[i,:,:] = stack[i,:,:]-np.amin(stack[i,:,:])
                stack[i,:,:] = int_range*stack[i,:,:]/np.amax(stack[i,:,:])         
                    
            stack = stack.astype('uint16')
            svPath = sLoc + '/' + str(sNum) + '.tif'
            io.imsave(svPath,stack)
                            
            sNum += 1