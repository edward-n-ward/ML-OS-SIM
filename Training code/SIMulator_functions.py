import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft, fft2, ifft2, fftshift, ifftshift

from skimage import io, transform
import scipy.special

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

    # Pattern OTF
    PSFp = calcPSF(512,opt.Psize,opt.NA,(opt.emission-30),1.518,1.33,opt.depthO,opt.depthO,1) # Excitation PSF  
    OTFp = np.abs(np.fft.fftshift(np.fft.fft2(PSFp)))


    # Illumination patterns
    frames = []
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            
            # Excitation pattern
            sig = (1 + opt.ModFac*cos(2*pi*(k2mat[i_a, 0]*(X) + k2mat[i_a, 1]*(Y)) + ps[i_a, i_s]))
            perfSig = (1 + 2*opt.ModFac*cos(2*pi*(k2mat[i_a, 0]*(X) + k2mat[i_a, 1]*(Y)) + ps[i_a, i_s]))

            sig_i = DIo*sig  # In focus fluorescent response
            perfSig = DIo*sig # Ideal SIM response
            
            sig_o = DOi*(1+opt.ModFac) # Out of focus fluorescent response
            

            focalPlane = np.fft.ifftshift(np.abs(ifft2(fft2(sig_i)*fftshift(OTFi)))) # In focus signal
            focalPlane = focalPlane/np.amax(focalPlane)
            
            outFocalPlane = np.fft.ifftshift(np.abs(ifft2(fft2(sig_o)*fftshift(OTFo)))) # Out of focus signal
            outFocalPlane = outFocalPlane/np.amax(outFocalPlane)
            
            ST = foreground*focalPlane + background*outFocalPlane # Total signal collected
            ST = ST/np.amax(ST)
            
            # Gaussian noise generation
            aNoise = opt.NoiseLevel/100  # noise
            nST = np.random.normal(0, aNoise*np.std(ST, ddof=1), (w, w))
            NoiseFrac = 1  # may be set to 0 to avoid noise addition

            # Poisson noise generation
            interim = ST*opt.Poisson
            if np.amax(interim)<400:
                poissonNoise = np.random.poisson(ST*opt.Poisson).astype(float)
                STnoisy = ST + NoiseFrac*nST + poissonNoise
            else:
                STnoisy = ST + NoiseFrac*nST
            
            frames.append(STnoisy)

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

    small = transform.resize(Io, (512, 512), anti_aliasing=True)

    w = small.shape[0]
    pixelSize = opt.Psize
    rindexObj = 1.518
    rindexSp = 1.2
    offset = opt.offset
    depthI = 10
    depthO = opt.depthO # Random depth of out of focus signal
    
    PSFi = calcPSF(w,pixelSize,opt.NA,opt.emission,rindexObj,rindexSp,depthI,0,offset) # In focus PSF
    PSFo = calcPSF(w,pixelSize,opt.NA,opt.emission,rindexObj,rindexSp,opt.depthO,depthO,offset) # Out of focus PSF    
    

    
    # To prevent aggressive background removal only add background to 90 percent of training data
    ratio = np.random.rand()
    if  ratio > 0.1:
        background = 0.6+0.3*np.random.rand() # Random background intensity
        foreground = 1
        fg = 1
    else:
        background = 0
        foreground = 1
        fg = 1
    
    small = transform.resize(Io, (512, 512), anti_aliasing=True)
    DIo = small.astype('float')
    DOi = Oi.astype('float')
                
    frames = SIMimages(opt, DIo, DOi, PSFi, PSFo, background, foreground)

    PSFi = calcPSF(w,pixelSize*2,opt.NA,opt.emission,rindexObj,rindexSp,depthI,0,offset)
    OTFi = np.abs(np.fft.fftshift(np.fft.fft2(PSFi)))
    GT = (np.abs(ifft2(fft2(DIo)*fftshift(OTFi)))) # In focus signal
    GT = GT/np.amax(GT)

    frames.append(PSFi)
    frames.append(GT)
    stack = np.array(frames)

    return stack

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