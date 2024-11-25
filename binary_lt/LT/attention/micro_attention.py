import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import kd_kl_loss

class micro_attention_factor_ver2(nn.Module):
    def __init__(self, gpu=None, num_classes=None):
        super(micro_attention_factor_ver2, self).__init__()
        self.gpu = None
        if gpu:
            self.gpu = gpu

        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(num_classes * 2, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )
        self.cls_num_list = None

    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits, reduction='none')
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option, reduction='none')
        inputs = torch.cat([logits, teacher_logits], dim=1)
        if self.gpu:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        factors = F.softmax(output, dim=1)

        total_loss = torch.stack([loss.mean(dim=1), balanced_loss.mean(dim=1)], dim=1)
        total_loss = total_loss * factors
        total_loss = total_loss.sum(dim=1).mean()

        return total_loss


class micro_attention_factor(nn.Module):
    def __init__(self, gpu=None, num_classes=None):
        super(micro_attention_factor, self).__init__()
        self.gpu = None
        if gpu:
            self.gpu = gpu

        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(num_classes * 2, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )
        self.cls_num_list = None

    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits, reduction='none')
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option, reduction='none')
        inputs = torch.cat([logits, teacher_logits], dim=1)
        if self.gpu:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        factors = F.softmax(output, dim=1)

        total_loss = torch.stack([loss.mean(dim=1), balanced_loss.mean(dim=1)], dim=1)
        total_loss = total_loss * factors
        total_loss = total_loss.sum(dim=1).mean()

        return total_loss