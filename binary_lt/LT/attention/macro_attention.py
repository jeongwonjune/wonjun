import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import kd_kl_loss


class macro_factor(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu_ver2(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu_ver2, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu_ver2_wider(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu_ver2_wider, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu_ver3(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu_ver3, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu_time(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu_time, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, time, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss, time])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu_time_ver2(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu_time_ver2, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, time, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss, time])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_relu_time_ver3(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_relu_time_ver3, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, time, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss, time])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_leakyrelu(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_leakyrelu, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_leakyrelu_ver2(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_leakyrelu_ver2, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
        )

        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

class macro_factor_leakyrelu_ver3(nn.Module):
    def __init__(self, gpu=None):
        super(macro_factor_leakyrelu_ver3, self).__init__()
        self.gpu = gpu
        self.loss_balancer = torch.nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 2),
        )
        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss

