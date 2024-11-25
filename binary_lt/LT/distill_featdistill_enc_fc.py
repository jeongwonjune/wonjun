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
from collections import OrderedDict
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

from dataset.imbalance_cifar import ImbalanceCIFAR10, ImbalanceCIFAR100
from dataset.imabalance_cub import Cub2011
from dataset.inat import INaturalist

from utils.utils import *
from utils.BalancedSoftmaxLoss import create_loss
from utils.feature_similarity import CosKLD, CosSched, transfer_conv

from models.resnet_imagenet import resnet152, resnet50, LearnableWeightScaling
from models.reactnet_imagenet import reactnet as reactnet_imagenet, Classifier
from attention.new_macro import get_attention_module

Dataloader = None
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cub', choices=['inat', 'cub', 'cifar10', 'cifar100', 'fgvc', 'dogs', 'cars', 'flowers'])
parser.add_argument('--imb_ratio', default=0.1, type=float)
parser.add_argument('--data', metavar='DIR', default='./data/')
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
parser.add_argument('--scheduler', default='cosann', type=str, choices=['cosann', 'lambda'])
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
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

parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--num_classes', default=200, type=int, help='num classes in dataset')

parser.add_argument('--use_ce', action='store_true', help='use PaCo supcon loss')
parser.add_argument('--no_kld_enc', action='store_true', help='use PaCo supcon loss')
parser.add_argument('--balance_ce', action='store_true', help='use balance ce')
parser.add_argument('--distill_T', default=2.0, type=float, help='value of Temperature of distillation')
parser.add_argument('--classwise_bound', action='store_true')
parser.add_argument('--lamda', default=[0.9, 0.7], type=float)
parser.add_argument('--fix_coef', action='store_true')
parser.add_argument('--fp16', action='store_true', help=' fp16 training')
parser.add_argument('--enc_loss', action='store_true')
parser.add_argument('--factor_ver', default='ver2', choices=['ver1', 'ver2', 'ver3', 'ver4', 'ver5'])
parser.add_argument('--factor_lr', default=0.001, type=float)
parser.add_argument('--use_time', action='store_true')
parser.add_argument('--use_time_norm', action='store_true')
parser.add_argument('--maximize', action='store_true')
best_acc = 0
best_head = 0
best_med = 0
best_tail = 0


def main():
    args = parser.parse_args()
    teacher_path = args.pretrained.split('moco_ckpt')[0]
    args.mark = '_'.join([
        ('epochs' + (str)(args.epochs)),
        ('lr' + (str)(args.lr)),
        ('bs' + (str)(args.batch_size)),
        ('factor_lr' + (str)(args.factor_lr)),
    ])
    classwise_bound = 'classwise_bound' if args.classwise_bound else 'no-classwise_bound'
    balance_distill = 'balance_distill_all'
    if args.use_time:
        use_time = 'use_time'
    elif args.use_time_norm:
        use_time = 'use_time_norm'
    else:
        use_time = 'no_time'
    args.root_model = os.path.join(teacher_path,
                                   'student',
                                   'feature_distill',
                                   balance_distill,
                                   classwise_bound,
                                   use_time,
                                   )
    if args.maximize:
        args.root_model = os.path.join(args.root_model, 'maximize')

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
    global best_head
    global best_med
    global best_tail
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
    if args.dataset == 'inat':
        args.cls_num_list = train_dataset.cls_num_list
    else:
        args.cls_num_list = train_dataset.get_cls_num_list()


    teacher_model = resnet152(num_classes=args.num_classes, return_features=True)
    if args.lws_model:
        lws_model = LearnableWeightScaling(num_classes=args.num_classes)
    else:
        lws_model = None

    #student_enc = reactnet_imagenet(num_classes=args.num_classes, ver2=True, return_feat=True)
    student_enc = reactnet_imagenet(num_classes=args.num_classes)
    student_fc = Classifier(num_classes=args.num_classes)

    feat_converter = transfer_conv(student_fc.fc.in_features, teacher_model.fc.in_features)

    use_time = args.use_time or args.use_time_norm
    if use_time:
        assert args.factor_ver != 'ver2'
    balance_factor = get_attention_module(version=args.factor_ver, use_time=use_time, gpu=args.gpu)
    balance_factor.set_cls_num_list(args.cls_num_list)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            student_enc.cuda(args.gpu)
            student_fc.cuda(args.gpu)
            teacher_model.cuda(args.gpu)
            feat_converter.cuda(args.gpu)
            balance_factor.cuda(args.gpu)
            if args.lws_model:
                lws_model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            student_enc = torch.nn.parallel.DistributedDataParallel(student_enc, device_ids=[args.gpu],
                                                                    find_unused_parameters=True)
            student_fc = torch.nn.parallel.DistributedDataParallel(student_fc, device_ids=[args.gpu],
                                                                   find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)
            feat_converter = torch.nn.parallel.DistributedDataParallel(feat_converter, device_ids=[args.gpu],
                                                                       find_unused_parameters=True)
            balance_factor = torch.nn.parallel.DistributedDataParallel(balance_factor, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
            if args.lws_model:
                lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)

            filename = f'{args.root_model}/moco_ckpt.pth.tar'
            if os.path.exists(filename):
                args.resume = filename

            if args.reload:
                state_dict = student_enc.state_dict()
                state_dict_ssp = torch.load(args.reload)['state_dict']

                print(state_dict_ssp.keys())

                for key in state_dict.keys():
                    print(key)
                    if key in state_dict_ssp.keys() and state_dict[key].shape == state_dict_ssp[key].shape:
                        state_dict[key] = state_dict_ssp[key]
                        print(key + " ****loaded******* ")
                    else:
                        print(key + " ****unloaded******* ")
                student_enc.load_state_dict(state_dict)

        else:
            student_enc.cuda()
            student_fc.cuda()
            teacher_model.cuda()
            feat_converter.cuda()
            balance_factor.cuda()
            if args.lws_model:
                lws_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            student_enc = torch.nn.parallel.DistributedDataParallel(student_enc, find_unused_parameters=True)
            student_fc = torch.nn.parallel.DistributedDataParallel(student_fc, find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, find_unused_parameters=True)
            feat_converter = torch.nn.parallel.DistributedDataParallel(feat_converter, find_unused_parameters=True)
            balance_factor = torch.nn.parallel.DistributedDataParallel(balance_factor, find_unused_parameters=True)
            if args.lws_model:
                lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student_enc = student_enc.cuda(args.gpu)
        student_fc = student_fc.cuda(args.gpu)
        teacher_model = teacher_model.cuda(args.gpu)
        feat_converter = feat_converter.cuda(args.gpu)
        balance_factor = balance_factor.cuda(args.gpu)
        if args.lws_model:
            lws_model = lws_model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in

        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    feat_loss = CosKLD(args=args)

    optimizer = torch.optim.Adam([
        {"params": student_enc.parameters()},
        {"params": student_fc.parameters()},
    ], args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    feat_conv_optimizer = torch.optim.Adam(feat_converter.parameters(), args.lr)
    feat_conv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=feat_conv_optimizer, T_max=args.epochs)

    balance_factor_optimizer = torch.optim.Adam(balance_factor.parameters(), args.factor_lr)
    balance_factor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=balance_factor_optimizer, T_max=args.epochs)

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

            enc_dict = checkpoint['enc_state_dict']
            fc_dict = checkpoint['fc_state_dict']
            balance_factor_dict = checkpoint['balance_factor']
            feat_conv_dict = checkpoint['feat_converter_state_dict']

            student_enc.load_state_dict(enc_dict)
            student_fc.load_state_dict(fc_dict)
            balance_factor.load_state_dict(balance_factor_dict)
            feat_converter.load_state_dict(feat_conv_dict)

            optimizer.load_state_dict(checkpoint['optimizer'])
            balance_factor_optimizer.load_state_dict(checkpoint['balance_factor_optimizer'])
            feat_conv_optimizer.load_state_dict(checkpoint['feat_conv_optimizer'])
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

    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.balance_ce:
        criterion = create_loss(cls_num_list=args.cls_num_list)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    feat_loss.set_cls_num_list(args.cls_num_list)
    if args.classwise_bound:
        class_weight = max(torch.Tensor(args.cls_num_list)) * (1 / torch.Tensor(args.cls_num_list))
        class_weight = class_weight.cuda()
    else:
        class_weight = None

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

    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, student_enc, student_fc, criterion_ce, logger, args)
        return

    # mixed precision
    scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        # print('5')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, teacher_model, lws_model, student_enc, student_fc,
              optimizer,
              feat_loss, class_weight, feat_converter, feat_conv_optimizer,
              balance_factor, balance_factor_optimizer,
              epoch, scaler, args)

        acc, loss, head_acc, med_acc, tail_acc = validate(val_loader, student_enc, student_fc, criterion_ce, logger, args)

        scheduler.step()
        feat_conv_scheduler.step()
        balance_factor_scheduler.step()

        print("Epoch: %d, Acc@1 %.3f" % (epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_head = head_acc
            best_med = med_acc
            best_tail = tail_acc
            is_best = True
        else:
            is_best = False
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'acc': acc,
                'enc_state_dict': student_enc.state_dict(),
                'fc_state_dict': student_fc.state_dict(),
                'balance_factor': balance_factor.state_dict(),
                'feat_converter_state_dict': feat_converter.state_dict(),

                'optimizer': optimizer.state_dict(),
                'balance_factor_optimizer': balance_factor_optimizer.state_dict(),
                'feat_conv_optimizer': feat_conv_optimizer.state_dict(),

            }, is_best=is_best, filename=f'{args.root_model}/ckpt.pth.tar')
            writer.add_scalar('val loss', loss, epoch)
            writer.add_scalar('val acc', acc, epoch)
            logger.info('Epoch: %d | Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f\n' % (epoch, best_acc, acc, loss))
    logger.info('Best Prec@1: %.2f %.2f %.2f %.2f ' % (best_acc, best_head, best_med, best_tail))


def train(train_loader, teacher_model, lws_model, student_enc, student_fc,
          optimizer,
          feat_loss, class_weight, feat_converter, feat_conv_optimizer,
          balance_factor, balance_factor_optimizer,
          epoch, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    kl_losses = AverageMeter('enc_kl', ':.4e')
    enc_feat_losses = AverageMeter('enc_feat', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, kl_losses, enc_feat_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student_enc.train()
    student_fc.train()
    teacher_model.eval()
    balance_factor.train()

    iters = len(train_loader)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            t_feat, t_out = teacher_model(images)
            if args.lws_model:
                t_out = lws_model(t_out)

        if not args.fp16:
            if hasattr(student_enc, "module"):
                student_enc.module.change_no_grad()
            else:
                student_enc.change_no_grad()
            s_feat, s_out = student_enc(images)
            s_feat = feat_converter(s_feat)
            s_out = student_fc(s_out)
            if args.use_time:
                iter = iters * epoch + i
                kl_loss = balance_factor(logits=s_out, teacher_logits=t_out, time=iter)
            elif args.use_time_norm:
                iter = iters * epoch + i
                iter /= iters * args.epochs
                kl_loss = balance_factor(logits=s_out, teacher_logits=t_out, time=iter)
            else:
                kl_loss = balance_factor(logits=s_out, teacher_logits=t_out)
            feature_loss = feat_loss(s_feat=s_feat, t_feat=t_feat, class_weight=class_weight, target=target)
            total_loss = 0.5 * kl_loss + 0.5 * feature_loss
            optimizer.zero_grad()
            balance_factor_optimizer.zero_grad()
            feat_conv_optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            feat_conv_optimizer.step()
            if args.maximize:
                for p in balance_factor.parameters():
                    p.grad = p.grad * -1
            balance_factor_optimizer.step()


        else:
            with autocast():
                if hasattr(student_enc, "module"):
                    student_enc.module.change_no_grad()
                else:
                    student_enc.change_no_grad()
                s_feat, feats = student_enc(images)
                s_feat = feat_converter(s_feat)
                s_out = student_fc(feats)

                if args.use_time:
                    iter = iters * epoch + i
                    kl_loss = balance_factor(logits=s_out, teacher_logits=t_out, time=iter)
                elif args.use_time_norm:
                    iter = iters * epoch + i
                    iter /= iters * args.epochs
                    kl_loss = balance_factor(logits=s_out, teacher_logits=t_out, time=iter)
                else:
                    kl_loss = balance_factor(logits=s_out, teacher_logits=t_out)
                feature_loss = feat_loss(s_feat=s_feat, t_feat=t_feat, class_weight=class_weight, target=target)
                total_loss = 0.5 * kl_loss + 0.5 * feature_loss

                optimizer.zero_grad()
                balance_factor_optimizer.zero_grad()
                feat_conv_optimizer.zero_grad()

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)

                balance_factor_optimizer.step()
                feat_conv_optimizer.step()
                scaler.update()

        acc1, acc5 = accuracy(s_out, target, topk=(1, 5))
        kl_losses.update(kl_loss.item() if kl_loss is not 0 else 0, s_out.size(0))
        enc_feat_losses.update(feature_loss.item() if feature_loss is not 0 else 0, s_out.size(0))

        top1.update(acc1[0], s_out.size(0))
        top5.update(acc5[0], s_out.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)


def validate(val_loader, model, classifier, criterion, logger, args):
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
    classifier.eval()
    total_s_out = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    class_num = torch.zeros(args.num_classes).cuda()
    correct = torch.zeros(args.num_classes).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output

            _, feat = model(images)
            output = classifier(feat)
            loss = criterion(output, target)

            total_s_out = torch.cat((total_s_out, output))
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