import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft, fft2, ifft2, fftshift, ifftshift

from skimage import io, transform
import scipy.special
from sqlalchemy import true

def save_image(arr):
    arr = arr/np.amax(arr)
    arr = 64000*arr
    arr = arr.astype('uint16')
    io.imsave('C:/Users/ew535/Documents/GitHub/VSR-SIM/data generation/pattern.tif',arr)

def SIMimages(opt, DIo, DOi, PSFi, PSFo, background,foreground):
    """
    Function to generate raw sim images
    
    INPUT VARIABLES:
       opt: contains all illumination variables
       DIo: specimen image
       DOi: out of focus image
       PSFi: in focus PSF
       PSFo: out of focus PSF
       background: strength of background
    
    OUTPUT VARIABLES:
       frames:  raw sim images
    """
    # Calculate both OTFs
    OTFi = np.fft.fftshift(np.fft.fft2(PSFi))
    OTFo = np.fft.fftshift(np.fft.fft2(PSFo))
    
    # Parameters for calculating illumination patterns
    w = DIo.shape[1]
    wo = w/2
    x = np.linspace(-wo*opt.Psize, wo*opt.Psize, w)
    y = np.linspace(-wo*opt.Psize, wo*opt.Psize, w)
    [X, Y] = np.meshgrid(x, y)

    # orientation direction of illumination patterns
    orientation = np.zeros(opt.Nangles)
    for i in range(opt.Nangles):
        orientation[i] = i*pi/opt.Nangles + opt.alpha + opt.angleError

    if opt.shuffleOrientations: 
        np.random.shuffle(orientation)

    # Illumination frequency vectors
    k2mat = np.zeros((opt.Nangles, 2))
    for i in range(opt.Nangles):
        theta = orientation[i]
        k2mat[i, :] = (opt.k2)*np.array([cos(theta), sin(theta)])

    # Illumination phase shifts along directions with errors
    ps = np.zeros((opt.Nangles, opt.Nshifts))
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            ps[i_a, i_s] = 2*pi*i_s/opt.Nshifts + opt.phaseError[i_a, i_s]



    # Illumination patterns
    frames = np.zeros((opt.Nangles*opt.Nshifts+1,w,w))
  
    index = 0
    for i_a in range(opt.Nangles):
        # centre = [w*np.random.rand(), w*np.random.rand()]
        # warp_factor = 0.02*np.random.rand()
        # rad = (np.random.rand()*opt.im_size/2)+500
        m = 1-np.random.rand()*0.4
        for i_s in range(opt.Nshifts):
            
            # Excitation pattern
            sig = (1 + m*cos(2*pi*(k2mat[i_a, 0]*(X) + k2mat[i_a, 1]*(Y)) + ps[i_a, i_s]))
            # sig = transform.swirl(sig, centre, strength=warp_factor, radius=rad, rotation=0, clip=True)
            # centre = [w*np.random.rand(), w*np.random.rand()]
            # warp_factor = 0.02*np.random.rand()
            # rad = (np.random.rand()*opt.im_size/2)+500
            # sig = transform.swirl(sig, centre, strength=warp_factor, radius=rad, rotation=0, clip=True)
            # save_image(sig)
            sig_i = DIo*sig  # In focus fluorescent response         
            sig_o = DOi*(1+opt.ModFac) # Out of focus fluorescent response
            

            focalPlane = np.fft.ifftshift(np.abs(ifft2(fft2(sig_i)*fftshift(OTFi)))) # In focus signal
            focalPlane = focalPlane/np.amax(focalPlane)
            
            outFocalPlane = np.fft.ifftshift(np.abs(ifft2(fft2(sig_o)*fftshift(OTFo)))) # Out of focus signal
            outFocalPlane = outFocalPlane/np.amax(outFocalPlane)
            
            ST = foreground*focalPlane + background*outFocalPlane # Total signal collected
            ST = ST/np.amax(ST)
                       
            frames[index,:,:] = ST
            index += 1
    frames[index,:,:] = sig
    return frames

def Generate_SIM_Image(opt, Io, Oi):
    """
    Run the SIMulation
    
    INPUT VARIABLES:
        opt: setup options
        Io: in focus image
        Oi: out of focus image
    
    OUTPUT VARIABLES:
        stack: stack of images used for network training
     
    """
    # PSF variables

    w = Io.shape[0]
    pixelSize = opt.Psize
    rindexObj = 1.52
    rindexSp = 1.36
    offset = opt.offset
    depthI = 10
    depthO = opt.depthO # Random depth of out of focus signal
    
    PSFi = calcPSF(w,pixelSize,opt.NA,opt.emission,rindexObj,rindexSp,depthI,0,offset) # In focus PSF
    PSFo = calcPSF(w,pixelSize,opt.NA,opt.emission,rindexObj,rindexSp,opt.depthO,depthO,offset) # Out of focus PSF    
    

    
    # To prevent aggressive background removal only add background to 50 percent of training data
    ratio = np.random.rand()
    if  ratio > 0.8:
        background = 0.3+0.2*np.random.rand() # Random background intensity
        foreground = 1
        fg = 1
    else:
        background = 0
        foreground = 1
        fg = 1
    
    DIo = Io.astype('float')
    DOi = Oi.astype('float')
    output = np.zeros((opt.Nangles*opt.Nshifts+2,w,w))
    frames = SIMimages(opt, DIo, DOi, PSFi, PSFo, background, foreground)
    # frames =transform.downscale_local_mean(frames,(1,2,2))
    # Apply noise to data
       
    for i in range(opt.Nangles*opt.Nshifts):
        ST = frames[i,:,:]
        aNoise = opt.NoiseLevel/100  # noise
        nST = np.random.normal(0, aNoise*np.std(ST, ddof=1), (ST.shape[0],ST.shape[1]))
        NoiseFrac = 0  # may be set to 0 to avoid noise addition

        # Poisson noise generation
        pFactor = opt.Poisson
        interim = ST*pFactor
        if np.amax(interim)<400:		
            poissonNoise = np.random.poisson(ST*pFactor).astype(float)
            STnoisy = ST + NoiseFrac*nST + poissonNoise
        else:
            STnoisy = ST + NoiseFrac*nST
        frames[i,:,:] = STnoisy

    if opt.pad == True:
        for i in range(opt.Nangles*opt.Nshifts):
            ST = np.fft.fftshift(np.fft.fft2(frames[i,:,:]))
            ST = np.pad(ST,[(128,128),(128,128)],mode = 'constant')
            ST = np.abs((np.fft.fft2(ST)))
            output[i,:,:] = ST[::-1,::-1]
            
    else:
        frames = transform.resize(frames, (opt.Nangles*opt.Nshifts+1,w,w),mode='edge',anti_aliasing=False,preserve_range=True,order=0)
        output[0:frames.shape[0],0:frames.shape[1],0:frames.shape[2]] = frames

    GT = DIo
    PSF_SIM = calcPSF(w,1.7*opt.Psize,opt.NA,opt.emission,rindexObj,rindexObj,0,0,offset) # Out of focus PSF
    OTF_SIM = np.power(np.abs(np.fft.fftshift(np.fft.fft2(PSF_SIM))),0.3) # flatten out the OTF equivalent to filtering reconstruction
   # GT = (np.abs(ifft2(fft2(GT)*fftshift(OTF_SIM)))) # In focus signal
    GT = GT/np.amax(GT)
    # output[frames.shape[0]-1,:,:] = OTF_SIM
    output[frames.shape[0],:,:] = GT

    return output

def calcPSF(xysize,pixelSize,NA,emission,rindexObj,rindexSp,depth,z_off,offset):
    """
    Generate the aberrated incoherent emission PSF using A.Stokseth model.
    Parameters
    
    INPUT VARIABLES:
    xysize: number of pixels
    pixelSize: size of pixels (in nm)
    emission: emission wavelength (in nm)
    rindexObj: refractive index of objective lens
    rindexSp: refractive index of the sample
    depth: imaging height above coverslip (in nm)
    z_off: distance from focal plane (in nm)
    offset: clip on max spatial frequency

    OUTPUT VARIABLES:
    psf: 2D array of incoherent PSF normalised between 0 and 1
    
    References
    ----------
    [1] P. A. Stokseth (1969), "Properties of a defocused optical system," J. Opt. Soc. Am. A 59:1314-1321. 
    """

    #Calculated the wavelength of light inside the objective lens and specimen
    lambdaObj = emission/rindexObj
    lambdaSp = emission/rindexSp

    #Calculate the wave vectors in vaccuum, objective and specimens
    k0 = 2*np.pi/emission
    kObj = 2*np.pi/lambdaObj
    kSp = 2*np.pi/lambdaSp

    #pixel size in frequency space
    dkxy = 2*np.pi/(pixelSize*xysize)

    #Radius of pupil
    kMax = offset*(2*np.pi*NA)/(emission*dkxy)

    klims = np.linspace(-xysize/2,xysize/2,xysize)
    kx, ky = np.meshgrid(klims,klims)
    k = np.hypot(kx,ky)
    pupil = np.copy(k)
    pupil[pupil<kMax]=1
    pupil[pupil>=kMax]=0

    #sin of objective semi-angle
    sinthetaObj = (k*(dkxy))/(kObj)
    sinthetaObj[sinthetaObj>1] = 1

    #cosin of objective semi-angle
    costhetaObj = np.finfo(float).eps+np.sqrt(1-(sinthetaObj**2))

    #sin of sample semi-angle
    sinthetaSp = (k*(dkxy))/(kSp)
    sinthetaSp[sinthetaSp>1] = 1

    #cosin of sample semi-angle
    costhetaSp = np.finfo(float).eps+np.sqrt(1-(sinthetaSp**2))

    #Defocus phase aberration
    phid = np.exp(1j*k0*costhetaObj*depth)
    
    #Spherical aberration phase calculation
    phisa = (1j*k0*depth)*((rindexSp*costhetaSp)-(rindexObj*costhetaObj))
    
    #Calculate the optical path difference due to spherical aberrations
    OPDSA = np.exp(phisa)

    #apodize the emission pupil
    pupil = (pupil/np.sqrt(costhetaObj))


    #calculate the aberrated pupil
    pupilSA = pupil*OPDSA*phid

    #calculate the coherent PSF
    psf = np.fft.ifft2(pupilSA)

    #calculate the incoherent PSF
    psf = np.fft.fftshift(abs(psf**2))
    

    return psf