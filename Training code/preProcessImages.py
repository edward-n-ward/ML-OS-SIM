from PIL import Image
import numpy as np
from skimage.exposure import match_histograms



def readTiff(path, n_images):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(n_images):
        print("Loading images: " + str(i+1) + "/" + str(n_images))
        try:
            img.seek(i)
            slice_ = np.zeros((img.height, img.width))
            for j in range(slice_.shape[0]):
                for k in range(slice_.shape[1]):
                    slice_[j,k] = img.getpixel((j, k))

            images.append(slice_)

        except EOFError:
            # Not enough frames in img
            break
    images = np.array(images)
    images = np.moveaxis(images, 0, -1)        
    

    for i in range(n_images):
        images[:,:,i] = images[:,:,i] - np.amin(images[:,:,i])
        images[:,:,i] = images[:,:,i] / np.amax(images[:,:,i])

    return (np.array((images)))

def imHistMatch(img1, img2):
    """
    img1 - reference image
    img2 - image to match
    """    
    img2 = match_histograms(img2, img1, multichannel=False)
    
    return img2

def removeBackground(img, frac):
    """
    img - image to correct
    frac - % of background to subtract
    """ 
    k = int(img.shape[0]*img.shape[1]*frac/100)
    value = np.partition(img.flatten(), k)[k]
    img = img - value
    img = img.clip(min=0)
    img = img/np.amax(img)
    
    return img

def calcPSF(xysize,pixelSize,NA,emission,rindexObj,rindexSp,depth):
    """
    Generate the aberrated emission PSF using P.A.Stokseth model.
    Parameters
    ----------
    xysize: number of pixels
    pixelSize: size of pixels (in nm)
    emission: emission wavelength (in nm)
    rindexObj: refractive index of objective lens
    rindexSp: refractive index of the sample
    depth: imaging height above coverslip (in nm)
    Returns
    -------
    psf:
        2D array of PSF normalised between 0 and 1
    
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
    kMax = (2*np.pi*NA)/(emission*dkxy)

    klims = np.linspace(-xysize/2,xysize/2,xysize)
    kx, ky = np.meshgrid(klims,klims)
    k = np.hypot(kx,ky)
    pupil = k
    pupil[pupil<kMax]=1
    pupil[pupil>=kMax]=0

    #sin of objective semi-angle
    sinthetaObj = (k*(dkxy))/kObj
    sinthetaObj[sinthetaObj>1] = 1

    #cosin of objective semi-angle
    costhetaObj = np.finfo(float).eps+np.sqrt(1-(sinthetaObj**2))

    #sin of sample semi-angle
    sinthetaSp = (k*(dkxy))/kSp
    sinthetaSp[sinthetaSp>1] = 1

    #cosin of sample semi-angle
    costhetaSp = np.finfo(float).eps+np.sqrt(1-(sinthetaSp**2))

    #Spherical aberration phase calculation
    phisa = (1j*k0*depth)*((rindexSp*costhetaSp)-(rindexObj*costhetaObj))
    #Calculate the optical path difference due to spherical aberrations
    OPDSA = np.exp(phisa)

    #apodize the emission pupil
    pupil = (pupil/np.sqrt(costhetaObj))


    #calculate the spherically aberrated pupil
    pupilSA = pupil*OPDSA

    #calculate the coherent PSF
    psf = np.fft.ifft2(pupilSA)**2

    #calculate the incoherent PSF
    psf = np.fft.fftshift(abs(psf))
    psf = psf/np.amax(psf)

    OTF = np.fft.fftshift(np.fft.fft2(psf))
    OTF = np.abs(OTF/np.amax(OTF))

    return psf, OTF

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

def quasiWeiner(img,OTF,noise_factor):
    
    abs_OTF=np.abs(OTF)
    conj_OTF=np.conj(OTF)
    
    ft=np.fft.fftshift(np.fft.fft2(img))
    filtered_ft=(ft*conj_OTF/(noise_factor+abs_OTF))
    img=np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_ft)))
    
    return img

def getFTProj(imgs):
    FT_imgs = np.zeros(imgs.shape)
    n_images = imgs.shape[2]
    for i in range(n_images):
        FT_imgs[:,:,i] = np.abs(np.fft.fftshift(np.fft.fft2(imgs[:,:,i])))

    FT_imgs = np.log(FT_imgs+1)
    FT_proj = FT_imgs.mean(axis=2)

    lims = np.linspace(-1,1,FT_proj.shape[0])
    x, y = np.meshgrid(lims,lims)
    rho = np.sqrt(x**2 + y**2) 
    FT_proj[rho<0.3] = 0

    FT_proj = FT_proj/np.amax(FT_proj)
    
    return FT_proj