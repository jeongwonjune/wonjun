import torch
import torchvision

import torch.nn as nn

model = nn.DataParallel(torchvision.models.resnet101()).cuda()

batch_size = 2400

data = torch.randn([batch_size, 3, 224, 224]).cuda()
model.eval()

with torch.no_grad():
    while True:
        output = model(data)
