import argparse
import builtins
import os

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import time
import json
import warnings
import logging
import pprint
import PIL
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

import moco.loader
import moco.builder
from utils.kd import statm_loss, transfer_conv

from dataset.imbalance_cifar import ImbalanceCIFAR10, ImbalanceCIFAR100
from dataset.imabalance_cub import Cub2011
from dataset.inat import INaturalist
from dataset.inat_moco import INaturalist_moco
from utils.autoaug import CIFAR10Policy, Cutout

from utils.BalancedSoftmaxLoss import create_loss
from utils.utils import *

from models.resnet_imagenet import resnet152, resnet50, LearnableWeightScaling
from models.reactnet_imagenet import reactnet as reactnet_imagenet

Dataloader = None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cub', choices=['inat', 'cub', 'cifar10', 'cifar100'])
parser.add_argument('--imb_ratio', default=0.1, type=float)
parser.add_argument('--data', metavar='DIR', default='./data/')
parser.add_argument('--root_path', type=str, default='./runs/renew/distill')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--teacher_model', help='choose model to use as a teacher', default='resnet152', type=str)
parser.add_argument('--pretrained', help='path for teacher model', metavar='PATH', type=str, default='')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--optimizer', help='choose which optimizer to use', default='adam', type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='warmup epochs')
parser.add_argument('--num_classes', default=200, type=int, help='num classes in dataset')
parser.add_argument('--feat_dim', default=1024, type=int,
                    help='last feature dim of backbone')
parser.add_argument('--kd_tech', default=None, type=str)
parser.add_argument('--kd_sub', default=None, type=str)

parser.add_argument('--alpha', default=0.05, type=float,
                    help='contrast weight among samples')
parser.add_argument('--beta', default=1.0, type=float,
                    help='contrast weight between centers and samples')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='paco loss')
parser.add_argument('--aug', default=None, type=str,
                    help='aug strategy')

parser.add_argument('--reverse_kldiv', action='store_true')
parser.add_argument('--losssrc2', action='store_true', help='use PaCo supcon loss')
parser.add_argument('--distill_T', default=2.0, type=float, help='value of Temperature of distillation')
parser.add_argument('--balance-distill', action='store_true', help='use balancing factor on distillation')
parser.add_argument('--balance-factor', default=1.0, type=float)
parser.add_argument('--balance-ce', action='store_true', help='use balance ce')

parser.add_argument('--fp16', action='store_true', help=' fp16 training')
parser.add_argument('--distill_option', type=str,
                    choices=['student', 'teacher_student'],
                    default='student')

best_acc = 0


def main():
    args = parser.parse_args()

    teacher_path = args.pretrained.split('moco_ckpt')[0]
    args.mark = '_'.join([
        ('epochs' + (str)(args.epochs)),
        ('lr' + (str)(args.lr)),
        ('bs' + (str)(args.batch_size)),
    ])
    distill_T = 'distill_T' + (str)(args.distill_T)

    args.root_model = os.path.join(teacher_path, 'student', distill_T)
    if args.balance_distill:
        args.root_model = os.path.join(args.root_model,
                                       ('model_balance_distill_' + args.distill_option + (str)(args.balance_factor)))
    use_loss = 'CE_' + (str)(args.losssrc2)
    if args.balance_ce:
        use_loss += '_balance'
    args.root_model = os.path.join(args.root_model, use_loss)

    if os.path.exists(args.root_model):
        trial = args.root_model.split('/')[-1]
        if 'try' in trial:
            try_num = (int)(trial[-2:])
            try_num += 1
            args.root_model = args.root_model + ('%2d' % (try_num))
        else:
            args.root_model = os.path.join(args.root_model, 'try01')

    os.makedirs(args.root_model, exist_ok=True)
    with open(os.path.join(args.root_model, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('exp save on: ', args.root_model)

    if 'mislas' in args.pretrained:
        args.lws_model = True
    else:
        args.lws_model = False

    if args.dist_url.endswith(':'):
        args.dist_url += (str)(np.random.randint(9000, 11000))

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
        args.lr *= args.batch_size / (ngpus_per_node * 128)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu

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
    if args.dataset == 'cub':
        args.num_classes = 200
        args.head_class_idx = [0, 72]
        args.med_class_idx = [72, 142]
        args.tail_class_idx = [142, 200]
    elif args.dataset == 'inat':
        args.num_classes = 8142
        args.head_class_idx = [0, 842]
        args.med_class_idx = [842, 4543]
        args.tail_class_idx = [4543, 8142]
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.head_class_idx = [0, 3]
        args.med_class_idx = [3, 7]
        args.tail_class_idx = [7, 10]
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.head_class_idx = [0, 36]
        args.med_class_idx = [36, 71]
        args.tail_class_idx = [71, 100]

    teacher_model = resnet152(num_classes=args.num_classes)
    if args.lws_model:
        lws_model = LearnableWeightScaling(num_classes=args.num_classes)
    else:
        lws_model = None

    student_model = reactnet_imagenet(num_classes=args.num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            student_model.cuda(args.gpu)
            teacher_model.cuda(args.gpu)
            if args.lws_model:
                lws_model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)
            if args.lws_model:
                lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)

            filename = f'{args.root_model}/moco_ckpt.pth.tar'
            if os.path.exists(filename):
                args.resume = filename

            if args.reload:
                state_dict = student_model.state_dict()
                state_dict_ssp = torch.load(args.reload)['state_dict']

                print(state_dict_ssp.keys())

                for key in state_dict.keys():
                    print(key)
                    if key in state_dict_ssp.keys() and state_dict[key].shape == state_dict_ssp[key].shape:
                        state_dict[key] = state_dict_ssp[key]
                        print(key + " ****loaded******* ")
                    else:
                        print(key + " ****unloaded******* ")
                student_model.load_state_dict(state_dict)

        else:
            student_model.cuda()
            teacher_model.cuda()
            if args.lws_model:
                lws_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            student_model = torch.nn.parallel.DistributedDataParallel(student_model, find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, find_unused_parameters=True)
            if args.lws_model:
                lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student_model = student_model.cuda(args.gpu)
        teacher_model = teacher_model.cuda(args.gpu)
        if args.lws_model:
            lws_model = lws_model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    kl_loss = kd_kl_loss
    optimizer = torch.optim.Adam(student_model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0)

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
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ### part to load teacher model
    assert os.path.isfile(args.pretrained)

    checkpoint = torch.load(args.pretrained)
    state_dict = checkpoint['state_dict']
    teacher_model.load_state_dict(state_dict)
    print('teacher weight load complete')
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    if args.lws_model:
        lws_dict = checkpoint['lws_dict']
        lws_model.load_state_dict(lws_dict)
        lws_model.eval()

    cudnn.benchmark = True

    transform_train_inat = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
    ])

    val_transform_inat = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
    ])
    transform_train_cub = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform_cub = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_train_cifar = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform_cifar = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if args.dataset == 'cub':
        args.data = os.path.join(args.data, 'cub')
        train_dataset = Cub2011(
            root=args.data,
            imb_type='exp',
            imb_factor=args.imb_ratio,
            train=True,
            transform=transform_train_cub
        )
        val_dataset = Cub2011(
            root=args.data,
            train=False,
            transform=val_transform_cub
        )
    elif args.dataset == 'inat':
        txt_train = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_train.txt'
        txt_test = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_val.txt'

        args.data = os.path.join(args.data, 'inat')
        train_dataset = INaturalist(
            root=args.data,
            txt=txt_train,
            transform=transform_train_inat
        )
        val_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=val_transform_inat
        )
    elif args.dataset == 'cifar10':
        train_dataset = ImbalanceCIFAR10(root=args.data, imb_type='exp', imb_factor=args.imb_ratio, rand_number=0,
                                         train=True, download=True, transform=transform_train_cifar)
        val_dataset = datasets.CIFAR10(
            root=args.data,
            train=False,
            download=True,
            transform=val_transform_cifar
        )

    elif args.dataset == 'cifar100':
        train_dataset = ImbalanceCIFAR100(root=args.data, imb_type='exp', imb_factor=args.imb_ratio, rand_number=0,
                                          train=True, download=True, transform=transform_train_cifar)
        val_dataset = datasets.CIFAR100(
            root=args.data,
            train=False,
            download=True,
            transform=val_transform_cifar
        )

    print(f'===> Training data length {len(train_dataset)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.balance_distill:
        if args.dataset == 'inat':
            args.cls_num_list = train_dataset.cls_num_list
        else:
            args.cls_num_list = train_dataset.get_cls_num_list()
    elif args.balance_ce:
        args.cls_num_list = train_dataset.get_cls_num_list()
    else:
        args.cls_num_list = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

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
    # logger.info('\n' + pprint.pformat(args))
    # logger.info('\n' + str(args))

    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.balance_ce:
        criterion = create_loss(cls_num_list=args.cls_num_list)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, train_loader, student_model, criterion_ce, args, logger)
        return
    # mixed precision
    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        # print('5')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, teacher_model, lws_model, student_model,
              criterion, kl_loss,
              optimizer, epoch, scaler, args)
        acc, loss = validate(val_loader, train_loader, student_model, criterion_ce, args, logger)
        scheduler.step()
        print("Epoch: %d, Acc@1 %.3f" % (epoch, acc))
        if acc > best_acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'acc': acc,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=f'{args.root_model}/moco_ckpt.pth.tar')
            writer.add_scalar('val loss', loss, epoch)
            writer.add_scalar('val acc', acc, epoch)
            logger.info('Epoch: %d | Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f\n' % (epoch, best_acc, acc, loss))
    logger.info('Best Prec@1: %.3f%%' % (best_acc))


def train(train_loader, teacher_model, lws_model, student_model,
          criterion_ce, kl_loss,
          optimizer, epoch, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student_model.train()
    teacher_model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        break
        loss_ce = None
        loss_paco = None
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            teacher_out = teacher_model(images)
            if args.lws_model:
                teacher_out = lws_model(teacher_out)
        if not args.fp16:
            logits = student_model(images)
            total_loss = kl_loss(logits, teacher_out, T=args.distill_t, cls_num_list=args.cls_num_list, option=args.distill_option)
        else:
            with autocast():
                logits = student_model(images)
                total_loss = kl_loss(logits, teacher_out, T=args.distill_t, cls_num_list=args.cls_num_list, option=args.distill_option)
        if args.losssrc2:
            total_loss *= 1 / 2
            total_loss += criterion_ce(logits, target) / 2
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(total_loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if not args.fp16:
            total_loss.backward()
            optimizer.step()

        else:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % 50 == 0:
        #     x = torch.arange(args.num_classes)
        #     plt.clf()
        #     a = teacher_out[0].squeeze().detach().cpu().numpy()
        #     b = logits[0].squeeze().detach().cpu().numpy()
        #     tmp_dir = os.path.join(args.root_model, 'stats', str(epoch))
        #     os.makedirs(tmp_dir, exist_ok=True)
        #     plt.plot(x, a, color='green', label='teacher')
        #     plt.plot(x, b, color='darkred', label='student')
        #     plt.fill_between(x, a, where=a >= 0, facecolor='lightgreen', interpolate=True)
        #     plt.fill_between(x, b, where=b >= 0, facecolor='lightcoral', interpolate=True)
        #     plt.legend(loc='best')
        #     plt.savefig(os.path.join(tmp_dir, f'{i}.png'))

        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(val_loader, train_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
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
