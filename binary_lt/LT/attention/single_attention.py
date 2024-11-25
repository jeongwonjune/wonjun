import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import kd_kl_loss

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
