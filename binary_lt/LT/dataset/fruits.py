import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms


class Fruits(VisionDataset):
    def __init__(self, root, train=True, extensions=None, transform=None,
                 target_transform=None, imb_factor=1, imb_type='exp', new_class_idx_sorted=None):
        root = os.path.join(root, 'fruits')
        if train:
            root = os.path.join(root, 'Training')
        else:
            root = os.path.join(root, 'Test')

        super(Fruits, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)

        categories = os.listdir(root)
        samples = []
        for c in categories:
            label = int(class_to_idx[c])
            image_path = os.listdir(os.path.join(root, c))
            for p in image_path:
                samples.append((os.path.join(root, c, p), label))

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.extensions = extensions
        self.classes = classes
        self.cls_num = len(classes)
        self.class_to_idx = class_to_idx

        self.samples = np.array([s[0] for s in samples])
        self.targets = np.array([s[1] for s in samples])

        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class

        self.sort_dataset(new_class_idx_sorted)
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)

    def sort_dataset(self, new_class_idx_sorted=None):
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.samples = self.samples[idx]
        if new_class_idx_sorted is None:
            new_class_idx_sorted = np.argsort(self.num_in_class)[::-1]
        for idx, target in enumerate(self.targets):
            self.targets[idx] = np.where(new_class_idx_sorted == target)[0]
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.samples = self.samples[idx]
        self.new_class_idx_sorted = new_class_idx_sorted

    def get_new_class_idx_sorted(self):
        return self.new_class_idx_sorted

    def __getitem__(self, index):
        # edit
        path = self.samples[index]
        target = self.targets[index]

        sample = Image.open(path)

        if self.transform:
            if isinstance(self.transform, list):
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample1 = self.transform[0](sample)
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample2 = self.transform[1](sample)
                sample = [sample1, sample2]
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # edit
        return sample, target

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = max(sorted(self.num_in_class)[::-1][1:])
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
        new_data = np.hstack(new_data)
        self.samples = new_data
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

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
    train_dataset = Fruits('/data', train=True, transform=train_transform, imb_factor=0.1)
    train_dataset.get_new_class_idx_sorted()
    test_dataset = Fruits('/data', train=False, transform=train_transform)
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
    classes_freq = np.zeros(24)
    for images, y in tqdm(train_loader):
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