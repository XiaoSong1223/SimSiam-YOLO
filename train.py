#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for Feature-Level SimSiam with YOLOv8 Backbone

import argparse
import os
import random
import shutil
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

# Import our modules
from my_experiment.simsiam_yolo import SimSiamYOLO, criterion


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Modified for small objects (traffic signs): reduced blur radius
    """

    def __init__(self, sigma=[.1, 1.0]):  # Reduced from [.1, 2.] for small objects
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class FlatImageDataset(Dataset):
    """
    兼容无子目录的纯图片文件夹，所有样本标签为 0。
    """

    def __init__(self, root, transform=None, extensions=None):
        self.root = root
        self.transform = transform
        if extensions is None:
            extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        self.samples = []
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(extensions):
                self.samples.append(fpath)
        if not self.samples:
            raise FileNotFoundError(f"No valid image files found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


parser = argparse.ArgumentParser(description='Feature-Level SimSiam Training with YOLOv8')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (unlabeled images)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--save-dir', default='./checkpoints', type=str,
                    help='directory to save checkpoints')

# SimSiamYOLO specific configs
parser.add_argument('--cfg', default='yolov8n.yaml', type=str,
                    help='YOLOv8 config file (default: yolov8n.yaml)')
parser.add_argument('--weights', default=None, type=str,
                    help='Path to pretrained YOLOv8 weights (optional)')
parser.add_argument('--dim', default=None, type=int,
                    help='Projector output dimension (default: same as input channels)')
parser.add_argument('--pred-dim', default=None, type=int,
                    help='Predictor hidden dimension (default: dim // 4)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--imgsz', default=640, type=int,
                    help='Input image size (default: 640)')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device (priority: CUDA > MPS > CPU)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using CUDA GPU: {args.gpu}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Create model
    print("=> creating model SimSiamYOLO")
    model = SimSiamYOLO(
        cfg=args.cfg,
        weights=args.weights,
        dim=args.dim,
        pred_dim=args.pred_dim,
        verbose=True
    )
    model = model.to(device)
    print(model)

    # Infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # Define optimizer
    if args.fix_pred_lr:
        optim_params = [
            {'params': model.encoder.parameters(), 'fix_lr': False},
            {'params': model.projector_p3.parameters(), 'fix_lr': False},
            {'params': model.projector_p4.parameters(), 'fix_lr': False},
            {'params': model.projector_p5.parameters(), 'fix_lr': False},
            {'params': model.predictor_p3.parameters(), 'fix_lr': True},
            {'params': model.predictor_p4.parameters(), 'fix_lr': True},
            {'params': model.predictor_p5.parameters(), 'fix_lr': True},
        ]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # For unlabeled data, ImageFolder will create a single class
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # Modified for small objects (traffic signs):
    # - RandomResizedCrop min scale: 0.2 (instead of 0.08) to preserve small objects
    # - GaussianBlur: reduced sigma range [.1, 1.0] (instead of [.1, 2.]) for small objects
    augmentation = [
        transforms.RandomResizedCrop(args.imgsz, scale=(0.2, 1.)),  # min scale 0.2 for small objects
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 1.0])], p=0.5),  # Reduced blur for small objects
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    base_transform = TwoCropsTransform(transforms.Compose(augmentation))

    # 优先尝试 ImageFolder（支持含类子目录的数据集），失败则回退到平铺图片目录
    try:
        train_dataset = datasets.ImageFolder(args.data, base_transform)
    except FileNotFoundError:
        print("ImageFolder 未找到类子目录，回退到纯图片目录加载方式")
        train_dataset = FlatImageDataset(args.data, base_transform)

    print(f"Dataset size: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, device, args)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, is_best=False, filename=os.path.join(args.save_dir, f'checkpoint_{epoch:04d}.pth.tar'))


def train(train_loader, model, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Move images to device
        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)

        # compute output and loss
        outputs = model(x1=images[0], x2=images[1])
        loss = criterion(outputs)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()

