import argparse
import builtins
import os
import random
import time
import warnings
import logging
import json
import pprint
from collections import OrderedDict

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
from models import resnet_cifar
from models import resnet_imagenet
import moco.loader
import moco.builder
from dataset.imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
from dataset.imabalance_cub import Cub2011
import torchvision.datasets as datasets
from utils.losses import PaCoLoss
from utils.autoaug import CIFAR10Policy, Cutout
from utils.utils import *

from models.reactnet_optimized import Reactnet as reactnet_cifar
from models.reactnet_imagenet import reactnet as reactnet_imagenet


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model_names += ['resnext101_32x4d']
model_names += ['resnet32']

parser = argparse.ArgumentParser()
parser.add_argument('--binary', action='store_true')
parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100', 'cub'])
parser.add_argument('--imb_ratio', default=0.1, type=float)
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./temp_runs/paco')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default='cosann', type=str, choices=['cosann', 'lambda', 'cosannwr'])
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--optimizer', help='choose which optimizer to use', default='sgd', type=str)
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=40, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')



# moco specific configs:
parser.add_argument('--moco-dim', default=64, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=1024, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.05, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=False, type=bool,
                    help='use cosine lr schedule')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='warmup epochs')
parser.add_argument('--alpha', default=0.02, type=float,
                    help='contrast weight among samples')
parser.add_argument('--beta', default=1.0, type=float,
                    help='supervise loss weight')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='paco loss')
parser.add_argument('--aug', default=None, type=str,
                    help='aug strategy')
parser.add_argument('--num_classes', default=100, type=int,
                    help='num classes in dataset')
parser.add_argument('--feat_dim', default=64, type=int,
                    help='last feature dim of backbone')
parser.add_argument('--use_normal_ce', action='store_true')


def main():
    args = parser.parse_args()

    if args.binary:
        args.root_path += '_binary'

    ce_type = 'paco_nomal_cifar' if args.use_normal_ce else 'paco_cifar'
    args.mark = '_'.join([
        ce_type,
        args.dataset,
        (str)(args.imb_ratio),
        ('epochs' + (str)(args.epochs)),
        ('lr' + (str)(args.lr)),
        ('bs' + (str)(args.batch_size)),
        args.optimizer
    ])

    args.root_model = os.path.join(args.root_path, args.mark)
    if not os.path.exists(args.root_model):
        os.makedirs(args.root_model, exist_ok=True)
    with open(os.path.join(args.root_model, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.head_class_idx = [0, 3]
        args.med_class_idx = [3, 7]
        args.tail_class_idx = [7, 10]
        input_size=32
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.head_class_idx = [0, 36]
        args.med_class_idx = [36, 71]
        args.tail_class_idx = [71, 100]
        input_size = 32
    elif args.dataset == 'cub':
        args.num_classes = 200
        args.head_class_idx = [0, 72]
        args.med_class_idx = [72, 142]
        args.tail_class_idx = [142, 200]
        input_size = 224

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.dataset == 'cub':
        if args.binary:
            base_model = reactnet_cifar
            args.feat_dim = 1024
        else:
            base_model = resnet_imagenet.resnet18
    if args.binary:
        base_model = reactnet_cifar
        args.feat_dim = 1024
    else:
        base_model = getattr(resnet_cifar, args.arch)
    if args.dataset == 'cub':
        model = moco.builder.MoCo(
            base_model,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.feat_dim, args.normalize,
            num_classes=args.num_classes)

    else:
        model = moco.builder.MoCo(
            base_model,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.feat_dim, args.normalize,
            num_classes=args.num_classes, cifar=True)
    print(model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = PaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k,
                         num_classes=args.num_classes).cuda(args.gpu)

    if args.binary:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs,
                                                               eta_min=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = None


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            new_dict = OrderedDict()
            for key in state_dict:
                new_dict[key.replace('module.', '')] = state_dict[key]
            del state_dict
            model.load_state_dict(new_dict)


            # model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    augmentation_regular = [
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = ImbalanceCIFAR10(root=args.data, imb_type='exp', imb_factor=args.imb_ratio, rand_number=0,
                                         train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(
            root=args.data,
            train=False,
            download=True,
            transform=val_transform)
    elif args.dataset == 'cifar100':
        train_dataset = ImbalanceCIFAR100(root=args.data, imb_type='exp', imb_factor=args.imb_ratio, rand_number=0,
                                          train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(
            root=args.data,
            train=False,
            download=True,
            transform=val_transform)
    elif args.dataset == 'cub':
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380],
                                         std=[0.195, 0.194, 0.192])

        augmentation_regular = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim)]

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        train_dataset = Cub2011(
            root=args.data,
            imb_type='exp',
            imb_factor=args.imb_ratio,
            rand_number=0,
            train=True,
            transform=transform_train)

        val_dataset = Cub2011(
            root=args.data,
            train=False,
            transform=val_transform)


    print(transform_train)

    print(f'===> Training data length {len(train_dataset)}')
    if args.use_normal_ce:
        criterion.cal_weight_for_classes(torch.ones(args.num_classes))
    else:
        criterion.cal_weight_for_classes(train_dataset.get_cls_num_list())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    log_dir = os.path.join(args.root_model, 'logs')
    writer = SummaryWriter(log_dir)
    log_file = os.path.join(log_dir, 'log_train.txt')
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, model, criterion_ce, args, logger)
        return

    best_acc1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        if not scheduler:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        if scheduler:
            scheduler.step()
        acc1, loss = validate(val_loader, model, criterion_ce, args, logger)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=f'{args.root_model}/moco_ckpt.pth.tar')
        if (epoch + 1) % args.print_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=f'{args.root_model}/moco_ckpt_{(epoch + 1):04d}.pth.tar')

        writer.add_scalar('val loss', loss, epoch)
        writer.add_scalar('val acc', acc1, epoch)
        logger.info('Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f\n' % (best_acc1, acc1, loss))
    logger.info('Best Prec@1: %.3f%%' % (best_acc1))
    writer.flush()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        features, labels, logits = model(im_q=images[0], im_k=images[1], labels=target)
        loss = criterion(features, labels, logits, epoch=epoch)

        total_logits = torch.cat((total_logits, logits))
        total_labels = torch.cat((total_labels, target))

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(args.num_classes).cuda()
    correct = torch.zeros(args.num_classes).cuda()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure head, tail classwise accuracy
            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, args.num_classes)
            predict_one_hot = F.one_hot(predicted, args.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)
        acc_classes = correct / class_num
        head_acc = acc_classes[args.head_class_idx[0]:args.head_class_idx[1]].mean() * 100

        med_acc = acc_classes[args.med_class_idx[0]:args.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[args.tail_class_idx[0]:args.tail_class_idx[1]].mean() * 100
        logger.info(
            '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(
                top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        # TODO: this should also be done with the ProgressMeter
        open(args.root_model + "/" + args.mark + ".log", "a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                                                                     .format(top1=top1, top5=top5))

    return top1.avg, losses.avg




if __name__ == '__main__':
    main()
