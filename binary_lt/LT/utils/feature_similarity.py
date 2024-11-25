import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosKLD(nn.Module):
    def __init__(self, size_average=False, args=None):
        super(CosKLD, self).__init__()
        self.size_average = size_average
        self.kldiv = nn.KLDivLoss(reduction='sum')
        # self.T = T
        self.mse = nn.MSELoss(reduction='none')
        # self.lamda = lamda
        # self.bkd_teacher = args.bkd_teacher
        # self.bkd_student = args.bkd_student
        # self.no_kld = args.no_kld

    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def init_weights(self, init_linear='normal'):
        pass

    def forward(self, s_feat, t_feat, class_weight=None, target=None):
        s_feat_flat = s_feat.view(s_feat.shape[0], -1)
        t_feat_flat = t_feat.view(t_feat.shape[0], -1)
        s_feat_norm = nn.functional.normalize(s_feat_flat, dim=1)
        t_feat_norm = nn.functional.normalize(t_feat_flat, dim=1)

        mse = self.mse(t_feat_norm, s_feat_norm)
        if class_weight is not None:
            mse = mse * class_weight[target][:, None]
        mse_sum = mse.sum()
        loss = mse_sum
        if self.size_average:
            loss /= s_feat.size(0)
        return loss

class CosSched:
    def __init__(self, args, loader):
        self.current_iter = 0
        self.total_iter = len(loader) * args.epochs
        self.lamda = args.lamda

    def step(self):
        self.current_iter += 1
        lamda = self.lamda[1] - (self.lamda[1] - self.lamda[0]) * (
                    (np.cos(np.pi * self.current_iter / self.total_iter)) + 1) / 2
        return lamda

class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature), nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, student):
        student = self.Connectors(student)
        return student