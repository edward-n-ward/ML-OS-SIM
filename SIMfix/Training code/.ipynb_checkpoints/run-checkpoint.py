import math
import os

import torch
import torch.nn as nn
import time

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import GetModel
from datahandler import GetDataloaders

from plotting import testAndMakeCombinedPlots

from options import parser

opt = parser.parse_args()

if opt.norm == '':
    opt.norm = opt.dataset
elif opt.norm.lower() == 'none':
    opt.norm = None

if len(opt.basedir) > 0:
    opt.root = opt.root.replace('basedir', opt.basedir)
    opt.weights = opt.weights.replace('basedir', opt.basedir)
    opt.out = opt.out.replace('basedir', opt.basedir)

if opt.out[:4] == 'root':
    opt.out = opt.out.replace('root', opt.root)


# convenience function
if len(opt.weights) > 0 and not os.path.isfile(opt.weights):
    # folder provided, trying to infer model options

    logfile = opt.weights + '/log.txt'
    opt.weights += '/final.pth'
    if not os.path.isfile(opt.weights):
        opt.weights = opt.weights.replace('final.pth', 'prelim.pth')

    if os.path.isfile(logfile):
        fid = open(logfile, 'r')
        optstr = fid.read()
        optlist = optstr.split(', ')

        def getopt(optname, typestr):
            opt_e = [e.split('=')[-1].strip("\'")
                     for e in optlist if (optname.split('.')[-1] + '=') in e]
            return eval(optname) if len(opt_e) == 0 else typestr(opt_e[0])

        opt.model = getopt('opt.model', str)
        opt.task = getopt('opt.task', str)
        opt.nch_in = getopt('opt.nch_in', int)
        opt.nch_out = getopt('opt.nch_out', int)
        opt.n_resgroups = getopt('opt.n_resgroups', int)
        opt.n_resblocks = getopt('opt.n_resblocks', int)
        opt.n_feats = getopt('opt.n_feats', int)


def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
            state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict


def train(dataloader, validloader, net, nepoch=10):

    start_epoch = 0
    if opt.task == 'segment':
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_function.cuda()
    if len(opt.weights) > 0:  # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint', opt.weights)
        if opt.undomulti:
            checkpoint['state_dict'] = remove_dataparallel_wrapper(
                checkpoint['state_dict'])
        if opt.modifyPretrainedModel:
            pretrained_dict = checkpoint['state_dict']
            model_dict = net.state_dict()
            # 1. filter out unnecessary keys
            for k, v in list(pretrained_dict.items()):
                print(k)
            pretrained_dict = {k: v for k, v in list(
                pretrained_dict.items())[:-2]}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(model_dict)

            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            net.load_state_dict(checkpoint['state_dict'])
            if opt.lr == 1:  # continue as it was
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']


        # if opt.modifyPretrainedModel:
        #     mod = list(net.children())
        #     mod.pop()
        #     mod.append(nn.Conv2d(64, 2, 1))
        #     net = torch.nn.Sequential(*mod)
        #     net.cuda()
        #     opt.task = 'segment'

    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        stepsize, gamma = int(opt.scheduler.split(
            ',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma)
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

    count = 0
    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, nepoch):
        mean_loss = 0

        for param_group in optimizer.param_groups:
            print('\nLearning rate', param_group['lr'])

        # if len(opt.lrseq) > 0:
        #     t = '[5,1e-4,10,1e-5]'
        #     t = np.array(t)
        #     epochvec = t[::2].astype('int')
        #     lrvec = t[1::2].astype('float')

        #     idx = epochvec.indexOf(epoch)
        #     opt.lr = lrvec[idx]
        #     optimizer = optim.Adam(net.parameters(), lr=opt.lr)

        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]

            optimizer.zero_grad()
            if opt.model == 'ffdnet':
                stdvec = torch.zeros(lr.shape[0])
                for j in range(lr.shape[0]):
                    noise = lr[j] - hr[j]
                    stdvec[j] = torch.std(noise)
                noise = net(lr.cuda(), stdvec.cuda())
                sr = torch.clamp(lr.cuda() - noise, 0, 1)
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            elif opt.task == 'residualdenoising':
                noise = net(lr.cuda())
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            else:
                sr = net(lr.cuda())
                if opt.task == 'segment':
                    if opt.nch_out > 2:
                        hr_classes = torch.round((opt.nch_out+1)*hr).long()
                        loss = loss_function(
                            sr.squeeze(), hr_classes.squeeze().cuda())
                    else:
                        loss = loss_function(
                            sr.squeeze(), hr.long().squeeze().cuda())
                else:
                    loss = loss_function(sr, hr.cuda())

            loss.backward()
            optimizer.step()

            ######### Status and display #########
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1, nepoch,
                                                    i+1, len(dataloader), loss.data.item()), end='')

            count += 1
            if opt.log and count*opt.batchSize // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                print(epoch, count*opt.batchSize, t1, mem,
                      mean_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0


        # ---------------- Scheduler -----------------
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate', param_group['lr'])
                break

        # ---------------- Printing -----------------
        print('\nEpoch %d done, %0.6f' %
              (epoch, (mean_loss / len(dataloader))))
        print('\nEpoch %d done, %0.6f' %
              (epoch, (mean_loss / len(dataloader))), file=opt.fid)
        opt.fid.flush()

        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net, validloader, opt, epoch)
            # if opt.scheduler:
            # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if len(opt.scheduler) > 0:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out, epoch+1))

    checkpoint = {'epoch': nepoch,
                  'state_dict': net.state_dict(),
                  'optimizer': optimizer.state_dict()}
    if len(opt.scheduler) > 0:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, opt.out + '/final.pth')


if __name__ == '__main__':

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    opt.fid = open(opt.out + '/log.txt', 'w')
    print(opt)
    print(opt, '\n', file=opt.fid)
    print('getting dataloader', opt.root)
    dataloader, validloader = GetDataloaders(opt)
    net = GetModel(opt)

    if opt.log:
        opt.train_stats = open(opt.out.replace(
            '\\', '/') + '/train_stats.csv', 'w')
        opt.test_stats = open(opt.out.replace(
            '\\', '/') + '/test_stats.csv', 'w')
        print('iter,nsample,time,memory,meanloss', file=opt.train_stats)
        print('iter,time,memory,psnr,ssim', file=opt.test_stats)

    import time
    t0 = time.perf_counter()
    if not opt.test:
        if opt.model.lower() == 'srgan':
            GANtrain(dataloader, validloader, net, nepoch=opt.nepoch)
        elif opt.model.lower() == 'esrgan':
            ESRGANtrain(dataloader, validloader, net, nepoch=opt.nepoch)
        else:
            train(dataloader, validloader, net, nepoch=opt.nepoch)
        # torch.save(net.state_dict(), opt.out + '/final.pth')
    else:
        if len(opt.weights) > 0:  # load previous weights?
            checkpoint = torch.load(opt.weights)
            print('loading checkpoint', opt.weights)
            if opt.undomulti:
                checkpoint['state_dict'] = remove_dataparallel_wrapper(
                    checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
            print('time: ', time.perf_counter()-t0)
        testAndMakeCombinedPlots(net, validloader, opt)
    print('time: ', time.perf_counter()-t0)
