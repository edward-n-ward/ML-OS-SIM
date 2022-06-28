import math
import numpy as np
import glob
from skimage import io, transform
import argparse
from SIMulator_functions import *

# ------------ normalisation-------------
def threshold_and_norm(arr):

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
    return arr

# ------------ Parameters-------------
def GetParams():
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 3
    # number of orientations of stripes
    opt.Nangles = 3
    # modulation factor
    opt.ModFac = 0.8+0.2*np.random.rand()
    # orientation offset
    opt.alpha = 0
    # orientation error
    opt.angleError = (0.5-np.random.rand())*2*np.pi
    # shuffle the order of orientations
    opt.shuffleOrientations = False
    # random phase shift errors
    opt.phaseError = 0*(0.5-np.random.rand(opt.Nangles, opt.Nshifts))
    # in percentage
    opt.NoiseLevel = 10 + 8*(np.random.rand()-0.5)
    # Poisson noise extent
    # include OTF and GT in stack
    opt.Poisson = np.random.randint(2000,10000)
    opt.OTF_and_GT = True
    # NA
    opt.NA = 1.49
    # Emission wavelength
    opt.emission = np.random.randint(500,690)
    # Pattern frequencies  
    opt.excitation = opt.emission-30
    kMax = (2*opt.NA)/(opt.emission)
    factor = 1 # Adjust for effects of aberrations
    opt.offset = factor
    opt.k2 = (2*opt.NA)/(opt.excitation)
    # Pixel Size
    opt.Psize = np.random.randint(100,105)
    # Image to use for ground truth in training
    opt.target = 'original' # 'original' or 'SIM' 
    # Out of focus depth
    opt.depthO = 700+800*np.random.rand()
    # Data size
    opt.im_size = 512 # Model data size
    opt.scale = 1 # Ground truth upscale
    opt.pad = False
    
    # Adjusted pixel Size
    opt.Psize = opt.Psize/2
    
    return opt


files = glob.glob('D:/ML-SIM/DIV2K/*.png')
sLoc = 'D:/ML-SIM/datasets/TIRF SIM'


n_rep = 2
sNum = 3191


for n_it in range(n_rep):
    n_it = n_it+1
   
    # ------------ Main loop --------------
    for f in range(len(files)):
        opt = GetParams()
        
        type = np.random.rand()
        if type > 0.1:
          # In focus image
          file = files[f]     
          Io = io.imread(file) 
          if  Io.shape[2] > 1:
            Io = Io.mean(2)
          Io = threshold_and_norm(Io)
          
          minDim = np.amin(Io.shape)
          # This is a very ugly way of re-sampling regions of the image
          Io = np.rot90(Io,np.random.randint(0,3))
          Io = Io[0:minDim,0:minDim]

          Io = transform.resize(Io, (opt.im_size*opt.scale, opt.im_size*opt.scale), anti_aliasing=True)
          Io[Io>1]=1
        
        elif type < 0.05:
          Io = np.zeros([opt.im_size*opt.scale,opt.im_size*opt.scale])
          row,col = Io.shape
          s_vs_p = 0.5
          amount = np.random.rand()*0.1
          # Salt mode
          num_salt = np.ceil(amount * Io.size * s_vs_p)
          coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in Io.shape]
          Io[coords] = 1
          file = files[f]
          Image = io.imread(file) 

          if  Image.shape[2] > 1:
            Image = Image.mean(2)
          Image = threshold_and_norm(Image)
          
          minDim = np.amin(Image.shape)
          # This is a very ugly way of re-sampling regions of the image
          Image = np.rot90(Image,np.random.randint(0,3))
          Image = Image[0:minDim,0:minDim]
          Image = transform.resize(Io, (opt.im_size*opt.scale, opt.im_size*opt.scale), anti_aliasing=True)
          Io = Image + Io



        else:
          Io = np.zeros([opt.im_size*opt.scale,opt.im_size*opt.scale])
          row,col = Io.shape
          s_vs_p = 0.5
          amount = np.random.rand()*0.1
          # Salt mode
          num_salt = np.ceil(amount * Io.size * s_vs_p)
          coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in Io.shape]
          Io[coords] = 1
                  
        # Out of focus image
        fnew = files[np.random.randint(0, len(files))]
        Oi = io.imread(fnew) 

        if  Oi.shape[2] > 1: # if not grayscale
            Oi = Oi.mean(2)  
            Oi = Oi/np.amax(Oi)

        Oi = threshold_and_norm(Oi)

        minDim = np.amin(Oi.shape)
        Oi = np.rot90(Oi,np.random.randint(0,3))
        Oi = Oi[0:minDim-1,0:minDim-1]
        Oi = transform.resize(Oi, (opt.im_size*opt.scale, opt.im_size*opt.scale), anti_aliasing=True)



        stack = Generate_SIM_Image(opt, Io, Oi)


        # normalise
        for i in range(len(stack)):
                stack[i,:,:] = stack[i,:,:]-np.amin(stack[i,:,:])
                stack[i,:,:] = 64000*stack[i,:,:]/np.amax(stack[i,:,:])         
                
        stack = stack.astype('uint16')
        svPath = sLoc + '/' + str(sNum) + '.tif'
        io.imsave(svPath,stack)
                        
        sNum += 1