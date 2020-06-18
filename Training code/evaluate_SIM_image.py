import math
import os

import torch
import time 

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import *
from datahandler import *

from skimage import io
import matplotlib.pyplot as plt
import glob
import argparse

def GetParams():
    
    opt = argparse.Namespace()
    opt.model='rcan'
    opt.lr=1e-4
    opt.norm='minmax'
    opt.nepoch=30
    opt.saveinterval=1
    opt.modifyPretrainedModel=''
    opt.multigpu=''
    opt.undomulti=''
    opt.ntrain=4000
    opt.scheduler=''
    opt.log='store_true'
    opt.noise=''

# data
    opt.dataset='fouriersim'
    opt.imageSize=512,
    opt.weights='C:/Users/ed_wa.DESKTOP-6TG1D20/Python work/ML python/Traning code/trained 17-06-2020/results/prelim.pth'
    opt.basedir='D:/Work/Test datasets'
    opt.root='D:/Work/Test datasets'
    opt.server=''
    opt.local='C:/Users/ed_wa.DESKTOP-6TG1D20/Python work/ML python/Traning code/trained 17-06-2020'
    opt.out='D:/Work/Test datasets/ML-SIM reconstructions/model 17-06-2020/'

    # computation 
    opt.workers=6
    opt.batchSize=1
    
    # restoration options
    opt.task='simin_gtout'
    opt.scale=1
    opt.nch_in=9
    opt.nch_out=1
    
    # architecture options 
    opt.narch=0 
    opt.n_resblocks=10
    opt.n_resgroups=3
    opt.reduction=4
    opt.n_feats=48

    # test options
    opt.ntest=10
    opt.testinterval=1
    opt.test=''
    opt.cpu=''
    opt.batchSize_test=6
    opt.plotinterval=5

    
    return opt

def loadimg(imgfile):
    stack = io.imread(imgfile)
    inputimgs,wfimgs = [],[]

    for i in range(int(len(stack)/9)):
        inputimg = stack[i*9:(i+1)*9]


        if opt.nch_in == 6:
            inputimg = inputimg[[0,1,3,4,6,7]]
        elif opt.nch_in == 3:
            inputimg = inputimg[[0,4,8]]
        
        if inputimg.shape[1] != 512 or inputimg.shape[2] != 512:
            print(imgfile,'not 512x512! Cropping')
            inputimg = inputimg[:,:512,:512]


        widefield = np.mean(inputimg,0)
        widefield = (widefield - np.min(widefield)) / (np.max(widefield) - np.min(widefield))    


        if opt.norm == 'convert': # raw img from microscope, needs normalisation and correct frame ordering
            print('Raw input assumed - converting')
            # NCHW
            # I = np.zeros((9,opt.imageSize,opt.imageSize),dtype='uint16')

            # for t in range(9):
            #     frame = inputimg[t]
            #     frame = 120 / np.max(frame) * frame
            #     frame = np.rot90(np.rot90(np.rot90(frame)))
            #     I[t,:,:] = frame
            # inputimg = I

            inputimg = np.rot90(inputimg,axes=(1,2))
            inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(inputimg)):
                inputimg[i] = 100 / np.max(inputimg[i]) * inputimg[i]
            inputimg = torch.tensor(inputimg.astype('float') / 255).float()
        elif 'convert' in opt.norm:
            fac = float(opt.norm[7:])
            inputimg = np.rot90(inputimg,axes=(1,2))
            inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(inputimg)):
                inputimg[i] = fac * 255 / np.max(inputimg[i]) * inputimg[i]
            inputimg = torch.tensor(inputimg.astype('float') / 255).float()
        elif opt.norm == 'minmax':
            inputimg = torch.tensor(inputimg.astype('float') / 255).float()
            for i in range(len(inputimg)):
                inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
            # inputimg = (inputimg - np.min(inputimg)) / (np.max(inputimg) - np.min(inputimg))
        elif 'minmax' in opt.norm:
            fac = float(opt.norm[6:])
            inputimg = torch.tensor(inputimg.astype('float') / 255).float()
            for i in range(len(inputimg)):
                inputimg[i] = fac * (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
        else:
            inputimg = torch.tensor(inputimg.astype('float') / 255).float()

        # otf = stack[9]
        # gt = stack[10]
        # simimg = stack[11] # sim

        

        # otf = torch.tensor(otf.astype('float') / np.max(otf)).unsqueeze(0).float()
        # gt = torch.tensor(gt.astype('float') / 255).unsqueeze(0).float()
        # simimg = torch.tensor(simimg.astype('float') / 255).unsqueeze(0).float()
        # widefield = torch.mean(inputimg,0).unsqueeze(0) 
        
        
        # normalise 
        # gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
        # simimg = (simimg - torch.min(simimg)) / (torch.max(simimg) - torch.min(simimg))
        # widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))    

        # save wf for reference
        widefield = torch.tensor(widefield).float()
        
        # io.imsave(opt.out + '/' + os.path.basename(imgfile).replace('.tif','_wf.png'),(255*widefield.numpy()).astype('uint8'))

        # save inputimg for reference - for debugging convertion part
        # io.imsave(opt.out + '/' + os.path.basename(imgfile), inputimg.numpy())    
        inputimgs.append(inputimg)
        wfimgs.append(widefield)

    return inputimgs,wfimgs

def EvaluateModel(opt):

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'\n',file=opt.fid)
    
    net = GetModel(opt)

    checkpoint = torch.load(opt.weights)
    if opt.cpu:
        net.cpu()
    print('loading checkpoint',opt.weights)
    net.load_state_dict(checkpoint['state_dict'])

    if opt.root.split('.')[-1] == 'tif':
        imgs = [opt.root]
    else:
        imgs = []
        imgs.extend(glob.glob(opt.root + '/*.tif'))
        if len(imgs) == 0: # scan everything
            imgs.extend(glob.glob(opt.root + '/**/*.tif',recursive=True))

    for i, imgfile in enumerate(imgs):

        inputimgs,wfimgs = loadimg(imgfile)
        frames = np.zeros([512,512,len(inputimgs)])
        wfs = np.zeros([512,512,len(inputimgs)])

        for j,(inputimg,wf) in enumerate(zip(inputimgs,wfimgs)):
            print('\r[%d/%d][%d/%d], shape is %dx%d - ' % (j+1,len(inputimgs),i+1,len(imgs),inputimg.shape[1],inputimg.shape[2]),end='')
            print('min max for',imgfile,inputimg.min(),inputimg.max())
            inputimg = inputimg.unsqueeze(0)

            with torch.no_grad():
                if opt.cpu:
                    sr = net(inputimg)
                else:
                    sr = net(inputimg.cuda())
                sr = sr.cpu()
                sr = torch.clamp(sr,min=0,max=1) 

                sr_frame = sr.numpy()
                sr_frame = np.squeeze(sr_frame)
                wf_frame = wf.numpy()
                wf_frame = np.squeeze(wf_frame)

                frames[:,:,j] = sr_frame
                wfs[:,:,j] = wf_frame

        frames = (frames * 255).astype('uint8')
        wfs = (wfs * 255).astype('uint8')
        if len(inputimgs) > 1:
            frames = np.moveaxis(frames,2,0)
            wfs = np.moveaxis(wfs,2,0)

        if opt.out == 'root': # save next to orignal
            svPath = imgfile[len(opt.root):-4] + '_sr.tif'
            io.imsave(svPath,frames)
            svPath = imgfile[len(opt.root):-4] + '_wf.tif'
            io.imsave(svPath,wfs)

        else:
            svPath = opt.out +imgfile[len(opt.root):-4] + '_sr.tif' 
            io.imsave(svPath,frames)
            svPath = opt.out +imgfile[len(opt.root):-4] + '_wf.tif'
            io.imsave(svPath,wfs)


if __name__ == '__main__':
    from options import parser
    opt = GetParams()

    if opt.norm == '':
        opt.norm = opt.dataset
    elif opt.norm.lower() == 'none':
        opt.norm = None

    if len(opt.basedir) > 0:
        opt.root = opt.root.replace('basedir',opt.basedir)
        opt.weights = opt.weights.replace('basedir',opt.basedir)
        opt.out = opt.out.replace('basedir',opt.basedir)
        
    EvaluateModel(opt)






            



