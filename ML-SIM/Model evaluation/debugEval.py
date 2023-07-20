import numpy as np
import glob
import sys
import os
import math
import os

import torch
import time 

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from skimage import io
from skimage.exposure import match_histograms
import tifffile
from models import *

from tqdm import tqdm


import argparse
def GetParams():
  opt = argparse.Namespace()

  # data
  opt.weights = 'C:/Users/ew535/Documents/GitHub/ML-OS-SIM/ML-SIM/Models/ML-SIM copy apo/prelim160.pth' # model to retrain from
  opt.imageSize = 512
  opt.root = 'C:/Users/ew535/University of Cambridge/User_CEB_LAG_MNG - ML for OS-SIM - ML for OS-SIM/Reconstruction error analysis/noise tests/dump'
  opt.out = 'C:/Users/ew535/University of Cambridge/User_CEB_LAG_MNG - ML for OS-SIM - ML for OS-SIM/Reconstruction error analysis/noise tests/dump/CNN' # output directory

  # input/output layer options
  opt.norm = 'modal' # if normalization should not be used
  opt.task = 'simin_gtout'
  opt.scale = 1
  opt.nch_in = 3
  opt.nch_out = 1

  # architecture options 
  opt.model='rcan'#'model to use'  
  opt.narch = 0
  opt.n_resgroups = 3
  opt.n_resblocks = 10
  opt.n_feats = 96
  opt.reduction = 16

  # test options
  opt.test = False
  opt.cpu = False # not supported for training
  opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
    
  return opt

def threshold_and_norm(arr,method='minmax'):
    """Experimental method for input image normalisation
	Args:
        arr: input array
        method: method to use for normalisation
	"""
    if method == 'minmax':
        arr = arr-np.amin(arr)
        arr = (arr/np.amax(arr))
    elif method == 'mean':
        arr = arr-np.mean(arr)
        arr = (arr/np.amax(arr))
    elif method == 'hist':
        hist, bins = np.histogram(arr,255)
        ind = np.where(hist==np.amax(hist))
        mini = bins[ind[0][0]]
        maxi = bins[ind[0][0]+1]
        sub = 0.8*((maxi+mini)/2)
        arr = arr - sub
        arr[arr<0]=0
        arr = (arr/np.amax(arr))
    return arr

def remove_dataparallel_wrapper(state_dict):
	"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary
	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict


def EvaluateModel(opt):

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'/n',file=opt.fid)
    
    net = GetModel(opt)
    print('loading checkpoint',opt.weights)
    checkpoint = torch.load(opt.weights,map_location=opt.device)

    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    net.module.load_state_dict(state_dict, strict=False)


    


    if opt.root.split('.')[-1] == 'png' or opt.root.split('.')[-1] == 'jpg':
        imgs = [opt.root]
    else:
        imgs = []
        imgs.extend(glob.glob(opt.root + '/*.jpg'))
        imgs.extend(glob.glob(opt.root + '/*.png'))
        imgs.extend(glob.glob(opt.root + '/*.tif'))
        if len(imgs) == 0: # scan everything
            imgs.extend(glob.glob(opt.root + '/**/*.jpg',recursive=True))
            imgs.extend(glob.glob(opt.root + '/**/*.png',recursive=True))
            imgs.extend(glob.glob(opt.root + '/**/*.tif',recursive=True))

    imageSize = opt.imageSize

    for i, imgfile in enumerate(imgs):
        description = 'Processing image [%d/%d]' % (i+1,len(imgs))
        images = io.imread(imgfile, plugin='pil')

        nImgs = images.shape[0] // opt.nch_in
        filename = os.path.basename(imgfile)[:-4]
        arr_max = np.amax(images)
        arr_min = np.amin(images)

        if opt.norm == 'total':
            images =  threshold_and_norm(images,method = 'minmax')

        for stack_idx in tqdm(range(nImgs),desc=description):
            stackSubset = images[stack_idx*opt.nch_in:(stack_idx+1)*opt.nch_in]
            #stackSubset = stackSubset-np.amin(stackSubset)
            stackSubset = stackSubset/np.amax(stackSubset)
            if opt.norm == 'total':
                stackSubset =  stackSubset - arr_min
                stackSubset = stackSubset/(arr_max-arr_min)
            else:
                for f in range(1,opt.nch_in):
                    stackSubset[f,:,:] =  threshold_and_norm(stackSubset[f,:,:])  
            wf = np.mean(stackSubset,0)
            sub_tensor = torch.from_numpy(stackSubset)
            sub_tensor = sub_tensor.unsqueeze(0)
            sub_tensor = sub_tensor.type(torch.FloatTensor)
            
            
            with torch.no_grad():
                if opt.cpu:
                    sr = net(sub_tensor)
                else:
                    sr = net(sub_tensor.cuda())
                sr = sr.cpu()

                sr = torch.clamp(sr[0],0,1)
                sr_frame = sr.numpy()
                sr_frame = np.squeeze(sr_frame)                               

            #if stack_idx == 0:
            #    reference = np.copy(sr_frame)
            #else: 
            #    sr_frame = match_histograms(sr_frame,reference)

        

            wf = (wf * arr_max).astype('uint16')
            svPath = opt.out + '/' + filename +'_wf.tif'
            tifffile.imsave(svPath,wf,append=True)


            sr_frame = (sr_frame * arr_max).astype('uint16')
            svPath = opt.out + '/' + filename +'_sr.tif'
            tifffile.imsave(svPath,sr_frame,append=True)


if __name__ == '__main__':
    opt = GetParams()

    EvaluateModel(opt)
