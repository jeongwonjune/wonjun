import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class Flowers(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, rand_number=0, imb_factor=1, imb_type='exp'):
        np.random.seed(rand_number)

        root = os.path.join(root, 'flowers')
        if train:
            excel_file = os.path.join(root, 'train.txt')
        else:
            excel_file = os.path.join(root, 'valid.txt')

        self.samples = pd.read_csv(excel_file, delimiter=' ')
        self.root_dir = root
        self.transform = transform
        self.targets = self.samples['TARGET'].array
        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets, dtype=np.int64)

        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class
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
        classes = np.unique(self.targets)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(self.targets == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.append(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
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
        img_path = os.path.join(self.root_dir, self.samples[index, 0])
        y_label = torch.tensor(self.samples[index, 1]).long()
        image = Image.open(img_path)
        if self.transform:
            if isinstance(self.transform, list):
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample1 = self.transform[0](image)
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample2 = self.transform[1](image)
                image = [sample1, sample2]
            else:
                image = self.transform(image)
        return (image, y_label)

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # train_dataset = Flowers(root='/data', train=True, download=False, transform=train_transform, imb_factor=1)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=1, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)
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
    #     plt.savefig(f'Flowers_{i}.png')
    train_dataset = Flowers('/data', train=True, download=False, transform=train_transform, imb_factor=0.1)
    test_dataset = Flowers('/data', train=False, download=False, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=128, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)
    # for images, y in train_loader:
    #     print(y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # classes_freq = np.zeros(102)
    # for x, y in tqdm.tqdm(train_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # classes_freq = np.zeros(102)
    # for x, y in tqdm.tqdm(test_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)

    # print(train_dataset.get_cls_num_list())

    mean = 0.
    std = 0.
    classes_freq = np.zeros(102)
    for images, y in train_loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(classes_freq)
    print(mean, std)


    # classes_freq = np.zeros(102)
    # for images, y in test_loader:
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)