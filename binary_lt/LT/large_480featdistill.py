import argparse
import builtins
import os

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import time
import json
import warnings
import logging

import torch.multiprocessing as mp
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter


from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

from dataset.dataset import get_dataset

from utils.utils import *
from utils.BalancedSoftmaxLoss import create_loss
from utils.feature_similarity import CosKLD, transfer_conv

from timm.models.efficientnet import tf_efficientnet_b7_ns, tf_efficientnetv2_m_in21k, LearnableWeightScaling
from models.reactnet_imagenet import reactnet as reactnet_imagenet, Classifier
from attention.new_macro import get_attention_module

# section for syncbatchnorm and cos scheduler
from utils.scheduler import GradualWarmupScheduler


Dataloader = None
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cub')
parser.add_argument('--data', metavar='DIR', default='./data/')
parser.add_argument('--imb_ratio', type=float, default=0.1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--teacher_model', help='choose model to use as a teacher', default='imgnet21k', type=str,
                    choices=['imgnet21k','jft300m'])
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
parser.add_argument('-d', '--balance_distill_layer', type=str, choices=['fc_only', 'enc_only', 'enc_fc', 'None'],
                    default='enc_only')
parser.add_argument('--classwise_bound', action='store_true')
parser.add_argument('--lamda', default=[0.9, 0.7], type=float)
parser.add_argument('--fix_coef', action='store_true')
parser.add_argument('--fp16', action='store_true', help=' fp16 training')
parser.add_argument('--enc_loss', action='store_true')
parser.add_argument('--factor_ver', default='ver3', choices=['ver1', 'ver2', 'ver3', 'ver4', 'ver5'])
parser.add_argument('--factor_lr', default=0.001, type=float)
parser.add_argument('--use_time', action='store_true')
parser.add_argument('--use_time_norm', action='store_true')
parser.add_argument('--maximize', action='store_true')

parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--no_pinmem', action='store_true')
parser.add_argument('--temp', type=float, default=2)

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
        args.scheduler,
        ('img_' + (str)(args.img_size)),
        ('temp_' + (str)(args.temp))
    ])
    classwise_bound = 'classwise_bound' if args.classwise_bound else 'no-classwise_bound'
    balance_distill = 'balance_distill_' + args.balance_distill_layer  # + args.distill_option
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
                                   args.mark,
                                   ('scaler' + (str)(args.fp16)),
                                   ('maximize' + (str)(args.maximize)),
                                   ('nopinmem' + (str)(args.no_pinmem))
                                   )

    os.makedirs(args.root_model, exist_ok=True)
    with open(os.path.join(args.root_model, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('exp save on: ', args.root_model)

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
        args.lr *= args.batch_size / 128
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

    if args.img_size == 480:
        train_img_size = (480, 224)
        val_img_size = 224
    elif args.img_size == 224:
        train_img_size = 224
        val_img_size = 224
    else:
        print('invalid img size')
        return

    train_dataset, val_dataset = get_dataset(args, train_img_size=train_img_size, val_img_size=val_img_size)

    print(f'===> Training data length {len(train_dataset)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    if args.dataset == 'inat':
        args.cls_num_list = train_dataset.cls_num_list
    else:
        args.cls_num_list = train_dataset.get_cls_num_list()

    if args.teacher_model == 'imgnet21k':
        teacher_model = tf_efficientnetv2_m_in21k(return_feat=True, num_classes=args.num_classes)
    elif args.teacher_model == 'jft300m':
        teacher_model = tf_efficientnet_b7_ns(return_feat=True, num_classes=args.num_classes)
    else:
        print('wrong pretrained teacher')
        return

    lws_model = LearnableWeightScaling(num_classes=args.num_classes)

    student_enc = reactnet_imagenet(num_classes=args.num_classes, ver2=True, return_feat=True)
    student_enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_enc)
    student_fc = Classifier(num_classes=args.num_classes)

    feat_converter = transfer_conv(student_fc.fc.in_features, teacher_model.classifier.in_features)

    use_time = (args.use_time or args.use_time_norm)
    if use_time:
        assert args.factor_ver != 'ver2'
    enc_factor = get_attention_module(version=args.factor_ver, use_time=use_time, gpu=args.gpu, temp=args.temp)
    fc_factor = get_attention_module(version=args.factor_ver, use_time=use_time, gpu=args.gpu, temp=args.temp)

    enc_factor.set_cls_num_list(args.cls_num_list)
    fc_factor.set_cls_num_list(args.cls_num_list)

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
            enc_factor.cuda(args.gpu)
            fc_factor.cuda(args.gpu)
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
            enc_factor = torch.nn.parallel.DistributedDataParallel(enc_factor, device_ids=[args.gpu],
                                                                   find_unused_parameters=True)
            fc_factor = torch.nn.parallel.DistributedDataParallel(fc_factor, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
            lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)

            filename = f'{args.root_model}/ckpt.pth.tar'
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
            fc_factor.cuda()
            enc_factor.cuda()
            lws_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            student_enc = torch.nn.parallel.DistributedDataParallel(student_enc, find_unused_parameters=True)
            student_fc = torch.nn.parallel.DistributedDataParallel(student_fc, find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, find_unused_parameters=True)
            feat_converter = torch.nn.parallel.DistributedDataParallel(feat_converter, find_unused_parameters=True)
            enc_factor = torch.nn.parallel.DistributedDataParallel(enc_factor, find_unused_parameters=True)
            fc_factor = torch.nn.parallel.DistributedDataParallel(fc_factor, find_unused_parameters=True)
            lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student_enc = student_enc.cuda(args.gpu)
        student_fc = student_fc.cuda(args.gpu)
        teacher_model = teacher_model.cuda(args.gpu)
        feat_converter = feat_converter.cuda(args.gpu)
        enc_factor = enc_factor.cuda(args.gpu)
        fc_factor = fc_factor.cuda(args.gpu)
        lws_model = lws_model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in

        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    feat_loss = CosKLD(args=args)

    enc_optimizer = torch.optim.Adam(student_enc.parameters(), args.lr)
    fc_optimizer = torch.optim.Adam(student_fc.parameters(), args.lr)

    enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=enc_optimizer, T_max=args.epochs)
    fc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=fc_optimizer, T_max=args.epochs)

    feat_conv_optimizer = torch.optim.Adam(feat_converter.parameters(), args.lr)
    feat_conv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=feat_conv_optimizer, T_max=args.epochs)

    fc_factor_optimizer = torch.optim.Adam(fc_factor.parameters(), args.factor_lr)
    enc_factor_optimizer = torch.optim.Adam(enc_factor.parameters(), args.factor_lr)
    fc_factor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=fc_factor_optimizer, T_max=args.epochs)
    enc_factor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=enc_factor_optimizer, T_max=args.epochs)



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

            student_enc.load_state_dict(checkpoint['enc_state_dict'])
            student_fc.load_state_dict(checkpoint['fc_state_dict'])
            enc_factor.load_state_dict(checkpoint['enc_factor'])
            fc_factor.load_state_dict(checkpoint['fc_factor'])
            feat_converter.load_state_dict(checkpoint['feat_converter_state_dict'])

            enc_optimizer.load_state_dict(checkpoint['enc_optimizer'])
            fc_optimizer.load_state_dict(checkpoint['fc_optimizer'])
            enc_factor_optimizer.load_state_dict(checkpoint['enc_factor_optimizer'])
            fc_factor_optimizer.load_state_dict(checkpoint['fc_factor_optimizer'])
            feat_conv_optimizer.load_state_dict(checkpoint['feat_conv_optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ### part to load teacher model
    assert os.path.isfile(args.pretrained)

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.pretrained, map_location=loc)

    state_dict = checkpoint['state_dict']
    teacher_model.load_state_dict(state_dict)
    print('teacher weight load complete')
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

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

    pin_mem = False if args.no_pinmem else True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=pin_mem, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_mem)

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
              enc_optimizer, fc_optimizer,
              criterion, kd_kl_loss,
              feat_loss, class_weight, feat_converter, feat_conv_optimizer,
              enc_factor, fc_factor, enc_factor_optimizer, fc_factor_optimizer,
              epoch, scaler, args)

        acc, loss, head_acc, med_acc, tail_acc = validate(val_loader, student_enc, student_fc, criterion_ce, logger,
                                                          args)

        enc_scheduler.step()
        fc_scheduler.step()
        feat_conv_scheduler.step()
        enc_factor_scheduler.step()
        fc_factor_scheduler.step()

        print("Epoch: %d, %.2f %.2f %.2f %.2f" % (epoch, acc, head_acc, med_acc, tail_acc))
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
                'enc_factor': enc_factor.state_dict(),
                'fc_factor': fc_factor.state_dict(),
                'feat_converter_state_dict': feat_converter.state_dict(),

                'enc_optimizer': enc_optimizer.state_dict(),
                'fc_optimizer': fc_optimizer.state_dict(),
                'enc_factor_optimizer': enc_factor_optimizer.state_dict(),
                'fc_factor_optimizer': fc_factor_optimizer.state_dict(),
                'feat_conv_optimizer': feat_conv_optimizer.state_dict(),

            }, is_best=is_best, filename=f'{args.root_model}/ckpt.pth.tar')
            writer.add_scalar('val loss', loss, epoch)
            writer.add_scalar('val acc', acc, epoch)
            logger.info('Epoch %d | Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f | time: %s\n' % (epoch, best_acc, acc, loss, time.asctime()))
    logger.info('Best Prec@1: %.2f %.2f %.2f %.2f ' % (best_acc, best_head, best_med, best_tail))
    open(args.root_model + "/" + "log.log", "a+").write('Best Prec@1: %.2f %.2f %.2f %.2f ' % (best_acc, best_head, best_med, best_tail))

def train(train_loader, teacher_model, lws_model, student_enc, student_fc,
          enc_optimizer, fc_optimizer,
          criterion, kl_loss,
          feat_loss, class_weight, feat_converter, feat_conv_optimizer,
          enc_factor, fc_factor, enc_factor_optimizer, fc_factor_optimizer,
          epoch, scaler, args, ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    enc_kl_losses = AverageMeter('enc_kl', ':.4e')
    enc_feat_losses = AverageMeter('enc_feat', ':.4e')

    fc_kl_losses = AverageMeter('fc_kl', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, enc_kl_losses, enc_feat_losses, fc_kl_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student_enc.train()
    student_fc.train()
    teacher_model.eval()
    lws_model.eval()
    fc_factor.train()
    enc_factor.train()

    iters = len(train_loader)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if args.teacher_model == 'imgnet21k':
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
            else:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            if args.teacher_model == 'imgnet21k':
                t_feat, t_out = teacher_model(images[0])
            else:
                t_feat, t_out = teacher_model(images)
            t_out = lws_model(t_out)

        if not args.fp16:
            if hasattr(student_enc, "module"):
                student_enc.module.change_no_grad()
            else:
                student_enc.change_no_grad()
            if args.teacher_model == 'imgnet21k':
                _, s_out = student_enc(images[1])
            else:
                _, s_out = student_enc(images)
            s_out = student_fc(s_out)

            if 'fc' in args.balance_distill_layer:
                if args.use_time:
                    iter = iters * epoch + i
                    fc_kl_loss = fc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                elif args.use_time_norm:
                    iter = iters * epoch + i
                    iter /= iters * args.epochs
                    fc_kl_loss = fc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                else:
                    fc_kl_loss = fc_factor(logits=s_out, teacher_logits=t_out)
            else:
                fc_kl_loss = kl_loss(s_out, t_out)

            fc_optimizer.zero_grad()
            fc_factor_optimizer.zero_grad()

            fc_kl_loss.backward()

            fc_optimizer.step()

            if args.maximize and ('fc' in args.balance_distill_layer):
                for p in fc_factor.parameters():
                    p.grad = p.grad * (-1)

            fc_factor_optimizer.step()

            if hasattr(student_enc, "module"):
                student_enc.module.change_with_grad()
            else:
                student_enc.change_with_grad()
            if args.teacher_model == 'imgnet21k':
                s_feat, s_out = student_enc(images[0])
            else:
                s_feat, s_out = student_enc(images)
            s_out = student_fc(s_out)
            s_feat = feat_converter(s_feat)

            enc_feat_loss = feat_loss(s_feat=s_feat, t_feat=t_feat, class_weight=class_weight, target=target)
            if 'enc' in args.balance_distill_layer:
                if args.use_time:
                    iter = iters * epoch + i
                    enc_kl_loss = enc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                elif args.use_time_norm:
                    iter = iters * epoch + i
                    iter /= iters * args.epochs
                    enc_kl_loss = enc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                else:
                    enc_kl_loss = enc_factor(logits=s_out, teacher_logits=t_out)
            else:
                enc_kl_loss = kl_loss(s_out, t_out)

            enc_loss = 0.5 * enc_feat_loss + 0.5 * enc_kl_loss

            enc_optimizer.zero_grad()
            feat_conv_optimizer.zero_grad()
            enc_factor_optimizer.zero_grad()

            enc_loss.backward()

            enc_optimizer.step()
            feat_conv_optimizer.step()
            if args.maximize and ('enc' in args.balance_distill_layer):
                for p in enc_factor.parameters():
                    p.grad = p.grad * (-1)
            enc_factor_optimizer.step()

        else:
            with autocast():
                if hasattr(student_enc, "module"):
                    student_enc.module.change_no_grad()
                else:
                    student_enc.change_no_grad()

                if args.teacher_model == 'imgnet21k':
                    _, s_out = student_enc(images[0])
                else:
                    _, s_out = student_enc(images)
                s_out = student_fc(s_out)

                if 'fc' in args.balance_distill_layer:
                    if args.use_time:
                        iter = iters * epoch + i
                        fc_kl_loss = fc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                    elif args.use_time_norm:
                        iter = iters * epoch + i
                        iter /= iters * args.epochs
                        fc_kl_loss = fc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                    else:
                        fc_kl_loss = fc_factor(logits=s_out, teacher_logits=t_out)
                else:
                    fc_kl_loss = kl_loss(s_out, t_out)
                fc_optimizer.zero_grad()
                fc_factor_optimizer.zero_grad()

                scaler.scale(fc_kl_loss).backward(retain_graph=True)

                scaler.step(fc_optimizer)
                scaler.update()
                if 'fc' in args.balance_distill_layer:
                    if args.maximize:
                        for p in fc_factor.parameters():
                            p.grad = p.grad * (-1)
                    scaler.step(fc_factor_optimizer)
                    scaler.update()

                if hasattr(student_enc, "module"):
                    student_enc.module.change_with_grad()
                else:
                    student_enc.change_with_grad()
                if args.teacher_model == 'imgnet21k':
                    s_feat, s_out = student_enc(images[0])
                else:
                    s_feat, s_out = student_enc(images)
                s_out = student_fc(s_out)
                s_feat = feat_converter(s_feat)

                enc_feat_loss = feat_loss(s_feat=s_feat, t_feat=t_feat, class_weight=class_weight, target=target)
                if 'enc' in args.balance_distill_layer:
                    if args.use_time:
                        iter = iters * epoch + i
                        enc_kl_loss = enc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                    elif args.use_time_norm:
                        iter = iters * epoch + i
                        iter /= iters * args.epochs
                        enc_kl_loss = enc_factor(logits=s_out, teacher_logits=t_out, time=iter)
                    else:
                        enc_kl_loss = enc_factor(logits=s_out, teacher_logits=t_out)
                else:
                    enc_kl_loss = kl_loss(s_out, t_out)
                enc_loss = 0.5 * enc_feat_loss + 0.5 * enc_kl_loss

                enc_optimizer.zero_grad()
                feat_conv_optimizer.zero_grad()
                enc_factor_optimizer.zero_grad()

                scaler.scale(enc_loss).backward()

                scaler.step(enc_optimizer)
                scaler.update()
                scaler.step(feat_conv_optimizer)
                scaler.update()
                if 'enc' in args.balance_distill_layer:
                    if args.maximize:
                        for p in enc_factor.parameters():
                            p.grad = p.grad * (-1)
                    scaler.step(enc_factor_optimizer)
                    scaler.update()

        acc1, acc5 = accuracy(s_out, target, topk=(1, 5))
        enc_kl_losses.update(enc_kl_loss.item() if enc_kl_loss != 0 else 0, s_out.size(0))
        enc_feat_losses.update(enc_feat_loss.item() if enc_feat_loss != 0 else 0, s_out.size(0))
        fc_kl_losses.update(fc_kl_loss.item() if fc_kl_loss != 0 else 0, s_out.size(0))

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
        open(args.root_model + "/" + "log.log", "a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                                                                     .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, head_acc, med_acc, tail_acc


if __name__ == '__main__':
    main()