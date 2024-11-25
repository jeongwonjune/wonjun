import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../')
from utils.utils import kd_kl_loss


def _make_layer(dimensions) -> nn.Sequential:
    layers = []
    for i in range(len(dimensions) - 1):
        layers.append(nn.Linear(in_features=dimensions[i], out_features=dimensions[i + 1]))
        layers.append(nn.ReLU())
    layers.pop()
    return nn.Sequential(*layers)

class macro_relu(nn.Module):
    def __init__(self, dimensions, gpu=None, temp=2):
        super(macro_relu, self).__init__()
        self.gpu = gpu
        self.attention_module = _make_layer(dimensions)
        self.factors = [1.0,0.0]
        self.temp = temp
    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, logits, teacher_logits, time=None, option='student'):

        loss = kd_kl_loss(logits, teacher_logits, T=self.temp)
        balanced_loss = kd_kl_loss(logits, teacher_logits, T=self.temp,
                                   cls_num_list=self.cls_num_list, option=option)
        if time is not None:
            inputs = torch.tensor([loss, balanced_loss, time])
        else:
            inputs = torch.tensor([loss, balanced_loss])

        if not self.gpu == None:
            inputs = inputs.cuda(self.gpu, non_blocking=True)

        output = self.attention_module(inputs)
        self.factors = F.softmax(output, dim=0)

        total_loss = self.factors[0] * loss + self.factors[1] * balanced_loss
        return total_loss


def get_attention_module(version='ver2', use_time=False, gpu=None, temp=2):
    if version == 'ver1':
        dimensions = [2, 8, 2]
    elif version == 'ver2':
        dimensions = [2, 8, 8, 2]
    elif version == 'ver3':
        dimensions = [2, 8, 16, 16, 8, 2]
    elif version == 'ver4':
        dimensions = [2, 16, 16, 2]
    elif version == 'ver5':
        dimensions = [2, 16, 32, 32, 16, 2]
    else:
        print('wrong option: ', version)
        assert False
    if use_time:
        dimensions[0] = 3
    return macro_relu(dimensions=dimensions, gpu=gpu, temp=temp)


if __name__ == "__main__":
    attention_model = get_attention_module(version='ver2', use_time='norm', gpu='cuda:0')
    print(attention_model)
