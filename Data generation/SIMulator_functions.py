import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft, fft2, ifft2, fftshift, ifftshift

from skimage import io, transform
import scipy.special

def SIMimages(opt, DIo, DOi, PSFi, PSFo, background):
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
    frames = []
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            
            # Excitation pattern
            sig = (1 + opt.ModFac*cos(2*pi*(k2mat[i_a, 0]*(X) + k2mat[i_a, 1]*(Y)) + ps[i_a, i_s]))
            
            # Defocused excitation pattern
            pat_o = np.fft.ifftshift(np.abs(ifft2(fft2(sig)*fftshift(OTFo))))
            pat_o = pat_o/np.amax(pat_o)

            sig_i = DIo*sig  # In focus fluorescent response
            sig_o = pat_o*DOi # Out of focus fluorescent response
            

            focalPlane = np.fft.ifftshift(np.abs(ifft2(fft2(sig_i)*fftshift(OTFi)))) # In focus signal
            outFocalPlane = np.fft.ifftshift(np.abs(ifft2(fft2(sig_o)*fftshift(OTFo)))) # Out of focus signal
            
            ST = focalPlane + background*outFocalPlane # Total signal collected
            
            # Gaussian noise generation
            aNoise = opt.NoiseLevel/100  # noise
            nST = np.random.normal(0, aNoise*np.std(ST, ddof=1), (w, w))
            NoiseFrac = 1  # may be set to 0 to avoid noise addition

            # Poisson noise generation
            poissonNoise = np.random.poisson(ST*opt.Poisson).astype(float)
            
            # noise added raw SIM images
            #STnoisy = ST + NoiseFrac*nST
            STnoisy = ST + NoiseFrac*nST + poissonNoise 
            frames.append(STnoisy)

    return frames

def drawGauss (std,width):
    """
    Generate a 2D Gaussian. Output is always square

    Parameters
    ----------
    std: Standard deviation of gaussian

    width: Width of square output

    Returns
    -------
    arg: Square array with 2D Gaussian function centred about the middle
    """

    width = np.linspace(-width/2,width/2,width) # Genate array of values around centre
    kx, ky = np.meshgrid(width,width) # Generate square arrays with co-ordinates about centre
    kx = np.multiply(kx,kx) # Calculate 2D Gaussian function
    ky = np.multiply(ky,ky)
    arg = np.add(kx,ky)
    arg = np.exp(arg/(2*std*std))
    arg = arg/np.sum(arg)
    arg = arg/np.amax(arg)

    return arg

def edgeTaper(I,PSF):
    """
    Taper the edge of an image with the provided point-spread function. This 
    helps to remove edging artefacts when performing deconvolution operations in 
    frequency space. The output is normalised  between [0,1] image with blurred 
    edges tapered to a value of 0.5.

    Parameters
    ----------
    I: Image to be tapered

    PSF: Point-spread function to be used for taper

        
    Returns
    -------
    tapered: Image with tapered edges

    """

    I = I/np.amax(I)

    PSFproj=np.sum(PSF, axis=0) # Calculate the 1D projection of the PSF
    # Generate 2 1D arrays with the tapered PSF at the leading edge
    beta1 = np.pad(PSFproj,(0,(I.shape[1]-1-PSFproj.shape[0])),'constant',constant_values=(0))
    beta2 = np.pad(PSFproj,(0,(I.shape[0]-1-PSFproj.shape[0])),'constant',constant_values=(0))

    # In frequency space replicate the tapered edge at both ends of each 1D array
    z1 = np.fft.fftn(beta1) # 1D Fourier transform 
    z1 = abs(np.multiply(z1,z1)) # Absolute value of the square of the Fourier transform
    z1=np.real(np.fft.ifftn(z1)) # Real value of the inverse Fourier transform
    z1 = np.append(z1,z1[0]) # Ensure the edges of the matrix are symetric 
    z1 = 1-(z1/np.amax(z1)) # Normalise

    z2 = np.fft.fftn(beta2)
    z2 = abs(np.multiply(z2,z2))
    z2=np.real(np.fft.ifftn(z2))
    z2 = np.append(z2,z2[0])
    z2 = 1-(z2/np.amax(z2))

    # Use matrix multiplication to generate a 2D edge filter
    q=np.matmul(z2[:,None],z1[None,:])

    #calculate the tapered image as the weighted sum of the blured and raw image
    tapered = np.multiply(I,q)+np.multiply((1-q),0.5*np.zeros(I.shape))
    Imax = np.amax(I)
    Imin = np.amin(I)

    # Bound the output by the min and max values of the oroginal image
    tapered[tapered < Imin] = Imin
    tapered[tapered > Imax] = Imax

    return tapered


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
    offset = 0.7+0.2*np.random.rand()
    depthI = 10
    depthO = 800+200*np.random.rand() # Random depth of out of focus signal
    
    PSFi = calcPSF(w,pixelSize,opt.NA,opt.emission,rindexObj,rindexSp,depthI,0,offset) # In focus PSF
    PSFo = calcPSF(w,pixelSize,opt.NA,opt.emission,rindexObj,rindexSp,depthO,depthO,offset) # Out of focus PSF    
    

    
    # To prevent aggressive background removal only add background to 60 percent of training data
    if np.random.rand() > 0.4:
        background = 0.7+0.1*np.random.rand() # Random background intensity
    else:
        background = 0
    
    
    if opt.target == 'original':

        Io = transform.resize(Io, (512, 512), anti_aliasing=True)
        DIo = Io.astype('float')
        DOi = Oi.astype('float')
                    
        frames = SIMimages(opt, DIo, DOi, PSFi, PSFo, background)

        frames.append(PSFi/np.amax(PSFi))
        frames.append(PSFo/np.amax(PSFo))
        frames.append(Io)
    
    else:

        

        small = transform.resize(Io, (512, 512), anti_aliasing=True)
        DIo = small.astype('float')
        DOi = Oi.astype('float')
                    
        frames = SIMimages(opt, DIo, DOi, PSFi, PSFo, background)
        frames.append(PSFi/np.amax(PSFi))

        OTFi = np.fft.fftshift(np.fft.fft2(PSFi))
        OTFsim = np.abs(np.pad(OTFi,((256,256),(256,256))))
        OTFtemp = np.abs(OTFsim)

        for a in range(opt.Nangles):

            theta = a*2*np.pi/opt.Nangles + opt.alpha + opt.angleError

            ky = int(512*opt.Psize*opt.k2*np.cos(theta))
            kx = int(512*opt.Psize*opt.k2*np.sin(theta))

            pos_shift = np.roll(OTFtemp,kx,axis=0)
            pos_shift = np.roll(pos_shift,ky,axis=1)

            neg_shift = np.roll(OTFtemp,-kx,axis=0)
            neg_shift = np.roll(neg_shift,-ky,axis=1)

            OTFsim = OTFsim+pos_shift+neg_shift

        OTFsim = OTFsim/np.amax(OTFsim)
        OTFsim[OTFsim>0.0001] = 1
        kernel = drawGauss (20,20)
        kernel = np.pad(kernel,((502,502),(502,502)))
        filter_f = np.fft.fft2(OTFsim)*np.fft.fft2(kernel)
        OTFsim = np.abs(np.fft.ifftshift(np.fft.ifft2(filter_f)))
        OTFsim = OTFsim/np.amax(OTFsim)

        Io_f =np.fft.fftshift(np.fft.fft2(Io))
        Io_f = Io_f*OTFsim
        Io = np.abs((np.fft.ifft2(np.fft.fftshift(Io_f))))
        Io = Io/np.amax(Io)


        gt11 = (Io[:512,:512])
        gt21 = (Io[512:,:512])
        gt12 = (Io[:512,512:])
        gt22 = (Io[512:,512:])


        OTFsim = transform.resize(OTFsim, (512, 512), anti_aliasing=True)
        frames.append(OTFsim)

        frames.append(gt11)
        frames.append(gt21)
        frames.append(gt12)
        frames.append(gt22)

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
    Returns
    z_off: distance from focal plane (in nm)
    
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