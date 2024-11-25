import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH


def eval_model():
    T = Trainer()
    T.dataset = 'cifar100-LT'
    T.resume = './ckpt/debug/cifar100-LT/vit_base_patch16/debug/checkpoint.pth'
    T.batch = 64
    T.device = '0'
    T.model = 'vit_base_patch16'
    T.input_size = 224
    T.global_pool = True
    T.num_workers = 16
    T.master_port = 29600
    T.evaluate()


eval_model()
