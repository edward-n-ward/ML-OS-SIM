import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import glob
from skimage import io
import random
import numpy as np

@DATASET_REGISTRY.register()
class upsampledSegmentDataset(data.Dataset):
    """use a patch to obtain low-res data image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def get_patch(self, lr, hr, patch_size=96, scale=4):
        iw, ih = lr.shape[2],lr.shape[1]
        p = 1
        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)


        tx, ty = scale * ix, scale * iy

        # lr = transforms.functional.crop(lr, iy, ix, ip, ip)
        # hr = transforms.functional.crop(hr, ty, tx, tp, tp)
        # lr = lr.crop((ix,iy,ix+ip,iy+ip))
        # hr = hr.crop((tx,ty,tx+tp,ty+tp))

        return [
            lr[:,iy:iy + ip, ix:ix + ip],
            hr[ty:ty + tp, tx:tx + tp]
        ]


    def __init__(self, opt):
        super(upsampledSegmentDataset, self).__init__()
        self.opt = opt

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        root = self.gt_folder
        self.images = []

        for folder in root.split(','):
            if ".tif" in folder:
                self.images.append(folder) # not a folder, but file (used for --test)
            else:
                folderimgs = sorted(glob.glob(folder + '/*.tif'))
                self.images.extend(folderimgs)

        random.seed(1234)
        random.shuffle(self.images)

        self.patchSize = opt['patchSize']
        self.scale = opt['scale']


        self.name = opt['name']

        self.len = len(self.images)


    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        path = self.images[index]
        stack = io.imread(path)

        if len(stack) > 9:
            scalefac = self.opt['scale']
            gt = stack[10] # used to be index self.nch_in+1
            nw = int(gt.shape[0]/scalefac) # I doubt it will work for non-squares
            nh = int(gt.shape[1]/scalefac) # but we'll try anyway

            inputimg = stack[0:9,0:nw,0:nh]

        else:
            gt = stack[0] # this will probably break everything
            inputimg = stack[4:5]

        # widefield = stack[12]

        # print('max before:',end=' ')
        # print('%0.2f %0.2f %0.2f %0.2f %0.2f' % (np.max(inputimg),np.max(otf),np.max(gt),np.max(simimg),np.max(widefield)))

        maxvalue = np.iinfo(inputimg.dtype).max
        inputimg = inputimg.astype('float')
        inputimg = inputimg /maxvalue
       
        gt = gt.astype('float') 
        gt = gt / maxvalue



        inputimg = torch.tensor(inputimg).float()
        gt = torch.tensor(gt).float()

        if self.patchSize is not None:
            inputimg, gt = self.get_patch(inputimg, gt, patch_size=self.patchSize, scale=self.scale)

        # if self.nch_out == 1:
            # gt = gt.unsqueeze(0)
        # widefield = torch.tensor(widefield).unsqueeze(0).float()
        # simimg = torch.tensor(simimg).unsqueeze(0).float()

        # normalise
        # gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
        # simimg = (simimg - torch.min(simimg)) / (torch.max(simimg) - torch.min(simimg))
        # widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))

        return {'lq': inputimg, 'gt': gt, 'lq_path': path, 'gt_path': path}

    def __len__(self):
        return len(self.images)
