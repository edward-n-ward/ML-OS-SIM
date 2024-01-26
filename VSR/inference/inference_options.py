# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from skimage import io,transform, exposure, img_as_float
import yaml
from collections import OrderedDict
import sys
from tqdm import tqdm
import tifffile
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/ew535.BLUE/Documents/GitHub/ML-OS-SIM/VSR-SIM')

from basicsr.models import build_model
from basicsr.archs.swinir_arch import SwinIR
from pathlib import Path

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
    type=str, 
    default='C:/Users/ew535.BLUE/University of Cambridge/User_CEB_LAG_MNG - ML for OS-SIM - ML for OS-SIM/Reconstruction error analysis/Confocal comparison/region 3/to process/', help='input test image folder')
    
    parser.add_argument('--output', 
    type=str, 
    default='C:/Users/ew535.BLUE/University of Cambridge/User_CEB_LAG_MNG - ML for OS-SIM - ML for OS-SIM/Reconstruction error analysis/Confocal comparison/region 3/to process/outputs/VSR/', help='output folder')
    
    parser.add_argument(
        '--task',
        type=str,
        default='simrec',
        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    # dn: denoising; car: compression artifact removal
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--patch_size', type=int, default=None, help='training patch size')
    parser.add_argument('--rolling', type=int, default=0, help='implement rolling SIM')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=0, help='noise level: 15, 25, 50')
    parser.add_argument('--in_chans', type=int, default=3, help='input channels: 1, 3, 9')    
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--large_model', action='store_true', help='Use large model, only used for real image sr')
    parser.add_argument(
        '--model_path',
        type=str,
        default='C:/Users/ew535.BLUE/Documents/GitHub/ML-OS-SIM/VSR-SIM/experiments/OS-SIM_tall/models/net_g_460000.pth')


    parser.add_argument(
        '-opt',
        type=str,
        default = 'C:/Users/ew535.BLUE/Documents/GitHub/ML-OS-SIM/VSR-SIM/experiments/OS-SIM_tall/VSR-SIM.yml',
        help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()


    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    opt['auto_resume'] = False
    opt['is_train'] = False
    opt['dist'] = False
    opt['num_gpu'] = 1
    opt['rank'], opt['world_size'] = 0,1

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    # model = define_model(args)

    model = build_model(opt).net_g # based on train script and options

    loadnet = torch.load(args.model_path)


    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    model.load_state_dict(loadnet[keyname], strict=False)

    model.eval()
    model = model.to(device)

    if args.task == 'jpeg_car':
        window_size = 7
    else:
        window_size = 4

    files = glob.glob(os.path.join(args.input, '*.tif'))

    files = sorted(files)


    for idx, path in enumerate(files):
       
        description = 'Processing image [%d/%d]' % (idx+1,len(files))
        # read image 
        imgname = os.path.splitext(os.path.basename(path))[0]

        # read image
        stack = io.imread(path, plugin='pil')
        stack = stack.astype(np.float32)
        if args.rolling == True:
            nImgs = stack.shape[0] - args.in_chans
        else:
            nImgs = stack.shape[0] // args.in_chans
        # stack = stack/255

        
        for stack_idx in tqdm(range(nImgs),desc=description):
            if args.rolling == True:
                inp = stack[stack_idx:stack_idx+args.in_chans]
            else:
                inp = stack[stack_idx*args.in_chans:(stack_idx+1)*args.in_chans]
                inp = inp[:,0:512,0:512]
                img_max = np.amax(inp)
                inp = threshold_and_norm(inp)

            img = torch.from_numpy(inp).float()
            img = img.unsqueeze(0).to(device)


            # inference
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = img.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                output = model(img)
                _, _, h, w = output.size()
                output = output[:, 0, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

            # save image
            os.makedirs(os.path.join(args.output, 'out'),exist_ok=True)
            svPath = os.path.join(args.output, f'out/{imgname}_out.tif')
            output = output.data.squeeze().float().cpu().clamp(0,1).numpy()
            output = (output*img_max).round().astype(np.uint16)
            # output = exposure.match_histograms(output, gt)
            # output = exposure.rescale_intensity(output,out_range=(0,1))
            # output = np.clip(output,(0,1))
            tifffile.imsave(svPath,output,append=True)
            #cv2.imwrite(os.path.join(args.output, f'out/{imgname}_out.png'), output)

            # wf
            os.makedirs(os.path.join(args.output, 'wf'),exist_ok=True)
            wf = np.mean(inp,axis=0)
            wf = transform.resize(wf,output.shape,order=3)
            svPath = os.path.join(args.output, f'wf/{imgname}_wf.tif')
            wf = (wf * img_max).round().astype(np.uint16)
            tifffile.imsave(svPath,wf,append=True)
            #cv2.imwrite(os.path.join(args.output, f'wf/{imgname}_wf.png'), wf)

if __name__ == '__main__':
    main()


