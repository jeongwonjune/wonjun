import argparse
import builtins
import os

import tqdm
from utils.scheduler import GradualWarmupScheduler
from utils.mislas import ClassAwareSampler, LabelAwareSmoothing, LearnableWeightScaling

from dataset.dataset import get_dataset

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import time
import warnings
import logging
import json
import PIL
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import timm

from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='flowers', choices=['inat', 'places','imagenet'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./runs/large_teacher')
parser.add_argument('--pretrained', type=str, default='imgnet21k', choices=['imgnet21k', 'jft300m'])
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default='cosannwr', type=str, choices=['cosann', 'lambda', 'cosannwr'])
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
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--num_classes', default=200, type=int,
                    help='num classes in dataset')
parser.add_argument('--fp16', action='store_true', help=' fp16 training')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--ckpt', default='', type=str)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--no_pinmem', action='store_true')
best_acc = 0


def main():
    args = parser.parse_args()
    if not args.dist_url:
        args.dist_url = 'tcp://localhost:' + (str)(np.random.randint(9000, 11000, 1)[0])

    args.mark = '_'.join([
        args.pretrained,
        ('epochs' + (str)(args.epochs)),
        ('lr' + (str)(args.lr)),
        ('bs' + (str)(args.batch_size)),
        args.optimizer,
        args.scheduler
    ])
    dataset = args.dataset + (str)(args.img_size)

    args.root_model = os.path.join(args.root_path, dataset, args.mark)
    if not args.fp16:
        args.root_model = os.path.join(args.root_model, 'no_scaler')
    if args.no_pinmem:
        args.root_model = os.path.join(args.root_model, 'no_pinmem')

    print('model save in: ', args.root_model)
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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        args.lr *= args.batch_size / 128
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    best_acc = 0
    args.gpu = gpu

    print(args)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

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

    train_dataset, val_dataset = get_dataset(args, train_img_size=args.img_size, val_img_size=args.img_size)

    if args.pretrained == 'imgnet21k':
        model_name = 'tf_efficientnetv2_m_in21k'
    elif args.pretrained == 'jft300m':
        model_name = 'tf_efficientnet_b7_ns'
    else:
        print('wrong pretrained model!')
        return
    model = timm.create_model(model_name, pretrained=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    fc_input_dim = model.classifier.in_features
    model.classifier = nn.Linear(fc_input_dim, args.num_classes)
    if not args.finetune:
        for param in model.parameters():
            param.requires_grad = False
            if len(param.shape) == 4:
                param.detach_()

    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    lws_model = None
    lws_model = LearnableWeightScaling(num_classes=args.num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if lws_model:
                lws_model.cuda(args.gpu)
                lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            filename = f'{args.root_model}/moco_ckpt.pth.tar'
            if os.path.exists(filename):
                args.resume = filename


        else:
            model.cuda()
            if lws_model:
                lws_model.cuda()
                lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, find_unused_parameters=True)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if lws_model:
            lws_model = lws_model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    parameters = [{'params': list(parameters)}]
    parameters.append({'params': lws_model.parameters()})

    scaler = GradScaler()
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True

    print(f'===> Training data length {len(train_dataset)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataset_ratio = None
    if args.dataset.lower() == 'cifar10':
        if args.imb_ratio == 0.1:
            smooth_head = 0.1
            smooth_tail = 0.0
        elif args.imb_ratio == 0.01:
            smooth_head = 0.3
            smooth_tail = 0.0
    elif args.dataset.lower() == 'cifar100':
        if args.imb_ratio == 0.1:
            smooth_head = 0.2
            smooth_tail = 0.0
        elif args.imb_ratio == 0.01:
            smooth_head = 0.4
            smooth_tail = 0.1
    elif args.dataset.lower() == 'inat':
        smooth_head = 0.3
        smooth_tail = 0.0
        dataset_ratio = 0.0537
    elif args.dataset.lower() == 'cub':
        smooth_head = 0.1
        smooth_tail = 0.0
    elif args.dataset.lower() == 'cars':
        smooth_head = 0.2
        smooth_tail = 0.0
    elif args.dataset.lower() == 'imagenet':
        smooth_head = 0.3
        smooth_tail = 0.0
    elif args.dataset.lower() == 'places':
        smooth_head = 0.4
        smooth_tail = 0.1
        dataset_ratio = 0.0343
    else:
        smooth_head = 0.1
        smooth_tail = 0.0
        print(smooth_head, smooth_tail)
    if not dataset_ratio:
        tmp_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=None,
            num_workers=args.workers, pin_memory=True, sampler=None)
        cls_num_list = train_dataset.get_cls_num_list()
        print('total classes:', len(cls_num_list))
        print('max classes:', max(cls_num_list))
        print('train loader length:', len(tmp_loader))
        train_loader_length = len(tmp_loader)
        oversampled_loader_length = len(cls_num_list) * max(cls_num_list)
        dataset_ratio = train_loader_length/oversampled_loader_length
        print('dataset ratio:', dataset_ratio)

    balance_sampler = ClassAwareSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=balance_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    args.epochs = int(dataset_ratio * args.epochs)
    if args.batch_size <= 128:
        if args.scheduler == 'cosann':
            print('cosann scheduler is used, no warmup')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'cosannwr':
            after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=after_scheduler)
            scheduler.step()
    else:
        print('batch size bigger than 127, using gradual warmup')
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=after_scheduler)
        scheduler.step()

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
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'].state_dict())
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define loss function (criterion) and optimizer


    criterion_ce = LabelAwareSmoothing(cls_num_list=train_dataset.get_cls_num_list(), smooth_head=smooth_head,
                                       smooth_tail=smooth_tail).cuda(args.gpu)


    log_dir = os.path.join(args.root_model, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir)
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


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch

        print('lr: ', get_lr(optimizer))
        train(train_loader, model,
              criterion_ce, optimizer,
              epoch, args,
              scaler, lws_model)
        acc, loss, head_acc, med_acc, tail_acc = validate(val_loader, model, criterion_ce, args, logger, lws_model)
        scheduler.step()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
                'lws_dict': lws_model.state_dict() if lws_model else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
                'scaler': scaler.state_dict()
            }, is_best=is_best, filename=f'{args.root_model}/moco_ckpt.pth.tar')

            # writer.add_scalar('val loss', loss, epoch)
            # writer.add_scalar('val acc', acc1, epoch)
            logger.info('Epoch %d | Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f | time: %s\n' % (epoch, best_acc, acc, loss, time.asctime()))
    logger.info('Best Prec@1: %.3f%%' % (best_acc))
    # writer.flush()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(train_loader, model,
          criterion, optimizer,
          epoch, args,
          scaler, lws_model):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()
    if lws_model:
        lws_model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            if args.batch_size > 2048:
                images = images.contiguous()
            target = target.cuda(args.gpu, non_blocking=True)
        #
        # compute output
        if not args.fp16:
            output = model(images)
            if lws_model:
                output = lws_model(output)
                # print(lws_model.module.learned_norm.data)
            loss = criterion(output, target)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with autocast():
                output = model(images)
                if lws_model:
                    output = lws_model(output)
                loss = criterion(output, target)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # measure elapsed time
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(acc1[0], output.size(0))
        top5.update(acc5[0], output.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(val_loader, model, criterion, args, logger, lws_model):
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
    if lws_model:
        lws_model.eval()
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
            if lws_model:
                output = lws_model(output)
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

    return top1.avg, losses.avg, head_acc, med_acc, tail_acc


if __name__ == '__main__':
    main()