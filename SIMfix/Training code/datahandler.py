import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageOps
import random

import numpy as np

from skimage import io
from skimage import exposure


def PSNR(I0,I1):
    MSE = torch.mean( (I0-I1)**2 )
    PSNR = 20*torch.log10(1/torch.sqrt(MSE))
    return PSNR


#normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                               std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(48),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

normalize2 = transforms.Normalize(mean = [0.69747254,0.53480325,0.68800158], std = [0.23605522,0.27857294,0.21456957])
unnormalize2 = transforms.Normalize(mean = [-2.9547, -1.9198, -3.20643], std = [4.2363, 3.58972, 4.66049])


toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      


def GetDataloaders(opt):

    if opt.dataset == 'SIMfix': 
        dataloader = load_SIMfix_dataset(opt.root,'train',opt)
        validloader = load_SIMfix_dataset(opt.root,'valid',opt)                
    else:
        print('unknown dataset')
        return None,None
    return dataloader, validloader


def load_image_dataset(root,category,opt):
        
    dataset = ImageDataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader


class DoubleImageDataset(Dataset):

    def __init__(self, root, category, opt):
        self.lq = glob.glob(root.split(',')[0] + '/**/*.tif', recursive=True)
        self.hq = glob.glob(root.split(',')[1] + '/**/*.tif', recursive=True)
        random.seed(1234)
        random.shuffle(self.lq)
        random.seed(1234)
        random.shuffle(self.hq)
        print(self.lq[:3])
        print(self.hq[:3])


        if category == 'train':
            self.lq = self.lq[:-10]
            self.hq = self.hq[:-10]
        else:
            self.lq = self.lq[-10:]
            self.hq = self.hq[-10:]

        self.imageSize = opt.imageSize
        self.scale = opt.scale
        self.nch_in = opt.nch_in
        self.len = len(self.lq)
        
    def __getitem__(self, index):
        with open(self.lq[index], 'rb') as f:
            img = Image.open(f)
            img = np.array(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            lq = Image.fromarray(img)
        with open(self.hq[index], 'rb') as f:
            img = Image.open(f)
            img = np.array(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            hq = Image.fromarray(img)
        
        # random crop
        w,h = lq.size
        ix = random.randrange(0,w-self.imageSize+1)
        iy = random.randrange(0,h-self.imageSize+1)

        lq = lq.crop((ix,iy,ix+self.imageSize,iy+self.imageSize))
        hq = hq.crop((ix,iy,ix+self.imageSize,iy+self.imageSize))

        lq, hq = toTensor(lq), toTensor(hq)
        
        # rotate and flip?
        if random.random() > 0.5:
            lq = lq.permute(0, 2, 1)
            hq = hq.permute(0, 2, 1)
        if random.random() > 0.5:
            lq = torch.flip(lq, [1])
            hq = torch.flip(hq, [1])
        if random.random() > 0.5:
            lq = torch.flip(lq, [2])
            hq = torch.flip(hq, [2])
                
        if self.nch_in == 1:
            lq = torch.mean(lq,0,keepdim=True)
            hq = torch.mean(hq,0,keepdim=True)
        
        return lq,hq,hq

    def __len__(self):
        return self.len


def load_doubleimage_dataset(root,category,opt):
        
    dataset = DoubleImageDataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader


class SIMfix_dataset(Dataset):

    def __init__(self, root, category, opt):

        self.images = []
        for folder in root.split(','):
            folderimgs = glob.glob(folder + '/*.tif')
            self.images.extend(folderimgs)

        random.seed(1234)
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.len = len(self.images)
        self.scale = opt.scale
        self.task = opt.task
        self.nch_in = opt.nch_in
        self.norm = opt.norm
        self.out = opt.out

    def __getitem__(self, index):
        
        stack = io.imread(self.images[index])
        inputimg = np.expand_dims(stack[:,:,0], axis=0)
        gt = stack[:,:,2]


        inputimg = inputimg.astype('float') / np.max(inputimg) # used to be /255
        gt = gt.astype('float') / np.max(gt) # used to be /255

        if self.norm == 'adapthist':
            for i in range(len(inputimg)):
                inputimg[i] = exposure.equalize_adapthist(inputimg[i],clip_limit=0.001)

            gt = exposure.equalize_adapthist(gt,clip_limit=0.001)


            inputimg = torch.tensor(inputimg).float()
            gt = torch.tensor(gt).unsqueeze(0).float()

        else:
            inputimg = torch.tensor(inputimg).float()
            gt = torch.tensor(gt).unsqueeze(0).float()

            # normalise 
            gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))

            if self.norm == 'minmax':
                for i in range(len(inputimg)):
                    inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
            elif 'minmax' in self.norm:
                fac = float(self.norm[6:])
                for i in range(len(inputimg)):
                    inputimg[i] = fac * (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
        
        return inputimg,gt
        
    def __len__(self):
        return self.len        

def load_SIMfix_dataset(root, category,opt):

    dataset = SIMfix_dataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader