import torch
import torch.nn as nn
import torch.nn.functional as F

class single_factor(nn.Module):
    def __init__(self, init_val=0.0):
        super(single_factor, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(init_val))
        self.sigmoid = nn.Sigmoid()
        self.cls_num_list = None

    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):
        factor = self.sigmoid(self.W)
        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        total_loss = (1 - factor) * loss + factor * balanced_loss
        return total_loss



class macro_attention(nn.Module):
    def __init__(self, base_encoder, gpu=None):
        super(macro_attention, self).__init__()
        self.gpu = gpu
        self.loss_balancer = base_encoder
        self.cls_num_list = None
        self.factors = None
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, option='student'):
        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits, cls_num_list=self.cls_num_list, option=option)
        inputs = torch.tensor([loss, balanced_loss])
        if self.gpu:
            inputs = inputs.cuda(self.gpu, non_blocking=True)
        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)
        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss



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


def kd_kl_loss(o_student, o_teacher, T=2, w=1, gamma=1, cls_num_list=None, option='teacher_student', reduction='mean'):
    if cls_num_list:
        cls_num_list = torch.Tensor(cls_num_list).view(1, len(cls_num_list))
        weight = cls_num_list / cls_num_list.sum()
        weight = weight.to(torch.device('cuda'))
        o_student += torch.log(weight + 1e-9) * gamma
        if option =='teacher_student':
            o_teacher += torch.log(weight + 1e-9) * gamma

    kl_loss = nn.KLDivLoss(reduction=reduction)(F.log_softmax(o_student / T, dim=1), F.softmax(o_teacher / T, dim=1)) * w
    # print(kl_loss)
    return kl_loss

class macro_attention_factor_leakyrelu_ver2(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_leakyrelu_ver2, self).__init__()
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


class macro_attention_factor_leakyrelu_ver3(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_leakyrelu_ver3, self).__init__()
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

class macro_attention_factor(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor, self).__init__()
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

class macro_attention_factor_relu(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_relu, self).__init__()
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

class macro_attention_factor_relu_ver2(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_relu_ver2, self).__init__()
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


class macro_attention_factor_relu_ver3(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_relu_ver3, self).__init__()
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



class macro_attention_factor_relu_time(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_relu_time, self).__init__()
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

class macro_attention_factor_relu_time_ver2(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_relu_time_ver2, self).__init__()
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

class macro_attention_factor_leakyrelu(nn.Module):
    def __init__(self, gpu=None):
        super(macro_attention_factor_leakyrelu, self).__init__()
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

    def forward(self, logits, teacher_logits, time=None, option='student'):

        loss = kd_kl_loss(logits, teacher_logits)
        balanced_loss = kd_kl_loss(logits, teacher_logits,
                                   cls_num_list=self.cls_num_list, option=option)
        if time:
            inputs = torch.tensor([loss, balanced_loss, time])
        else:
            inputs = torch.tensor([loss, balanced_loss])
        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.loss_balancer(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss


def get_macro(args):

    base_encoder = torch.nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    return macro_attention(base_encoder=base_encoder, gpu=args.gpu)