import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class DTD(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, rand_number=0, imb_factor=0.1, imb_type='exp'):
        np.random.seed(rand_number)
        self.root_dir = os.path.join(root, 'dtd')
        self.categories = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
        if train:
            excel_file = [os.path.join(self.root_dir, 'labels', 'train1.txt')]
            excel_file += [os.path.join(self.root_dir, 'labels', 'val1.txt')]
        else:
            excel_file = os.path.join(self.root_dir, 'labels', 'test1.txt')
        self.samples = []
        if isinstance(excel_file, list):
            for file in excel_file:
                self.samples += list(pd.read_csv(file)['PATH'])
        else:
            self.samples = list(pd.read_csv(excel_file)['PATH'])

        self.transform = transform
        self.targets = []
        for s in self.samples:
            class_name = s.split('/')[0]
            self.targets.append(self.categories.index(class_name))

        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class

        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)

        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        self.samples = np.array(self.samples)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.extend(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.array(new_data)
        self.samples = new_data
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, 'images', self.samples[index]))

        target = self.targets[index]
        target = torch.tensor(target).long()

        if self.transform is not None:
            if isinstance(self.transform, list):
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample1 = self.transform[0](img)
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample2 = self.transform[1](img)
                img = [sample1, sample2]
            else:
                img = self.transform(img)
        if isinstance(img, list):
            if img[0].shape == torch.Size([1, 224, 224]):
                img[0] = img[0].repeat(3, 1, 1)

            if img[1].shape == torch.Size([1, 224, 224]):
                img[1] = img[1].repeat(3, 1, 1)
        else:
            if img.shape == torch.Size([1, 224, 224]):
                img = img.repeat(3, 1, 1)

        return img, target


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = DTD(root='/data', train=True, download=False, transform=train_transform, imb_factor=0.1)
    test_dataset = DTD(root='/data', train=False, download=False, transform=train_transform)

    # print(train_dataset.get_cls_num_list())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # for i in range(len(train_dataset.get_cls_num_list())):
    #     images = torch.empty(train_dataset.get_cls_num_list()[0], 3, 224, 224)
    #     idx = 0
    #     for image, y in train_loader:
    #         if y == i:
    #             images[idx] = image
    #             idx += 1
    #
    #     plt.figure()
    #     plt.title(f'{i}')
    #     plt.clf()
    #     plt.imshow(torchvision.utils.make_grid(images, normalize=True).permute(1, 2, 0))
    #     plt.savefig(f'DTD_{i}.png')

    # classes_freq = np.zeros(47)
    # for x, y in tqdm.tqdm(test_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)

    mean = 0.
    std = 0.
    classes_freq = np.zeros(47)
    for images, y in tqdm.tqdm(train_loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(classes_freq)
    print(mean, std)
