# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore.nn as nn
from mindspore.ops import operations as P
from .binarylib import AdaBinConv2d, Maxout
import torch.nn.init as init

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2


# LambdaLayer class definition
class LambdaLayer(nn.Cell):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out
    
# BasicBlock class definition
class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = AdaBinConv2d(in_planes, planes, kernel_size=3, stride=stride, pad_mode="pad", padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = Maxout(planes)

        self.conv2 = AdaBinConv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = Maxout(planes)

        self.pad = nn.SequentialCell()
        if stride != 1 or in_planes != planes:
            self.pad = nn.Pad(((0, 0), (planes // 4, planes // 4), (0, 0), (0, 0)))

    def construct(self, x):
        out = self.bn1(self.conv1(x))
        if self.stride != 1 or self.in_planes != self.planes:
            x = x[:, :, ::2, ::2]
        out += self.pad(x)
        out = self.nonlinear1(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = self.nonlinear2(out)
        return out


class adabin(nn.Module):
    def __init__(self, num_classes=1000):
        super(adabin, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, feat_in=1024, num_classes=1000):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes)
        init.kaiming_normal_(self.fc.weight)
    def forward(self, x):
        x = self.fc(x)
        return x