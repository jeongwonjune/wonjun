import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


class INaturalist(Dataset):
    num_classes = 8142
    def __init__(self, root, txt, transform=None, new_class_idx_sorted=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        self.img_path = np.array(self.img_path)
        self.targets = np.array(self.targets)
        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class

        # self.sort_dataset(new_class_idx_sorted)

        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

    def sort_dataset(self, new_class_idx_sorted=None):
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.img_path = self.img_path[idx]
        if new_class_idx_sorted is None:
            new_class_idx_sorted = np.argsort(self.num_in_class)[::-1]
        for idx, target in enumerate(self.targets):
            self.targets[idx] = np.where(new_class_idx_sorted == target)[0]
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.img_path = self.img_path[idx]
        self.new_class_idx_sorted = new_class_idx_sorted

    def get_cls_num_list(self):
        return self.cls_num_list

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
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

        return sample, label  # , index

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Resize(32),
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

    txt_train = os.path.join('/data', 'iNaturalist18_train.txt')
    txt_test = os.path.join('/data', 'iNaturalist18_val.txt')
    train_dataset = INaturalist('/data', txt=False, transform=train_transform)
    test_dataset = INaturalist('/data',  txt=False, transform=train_transform)
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

    # mean = 0.
    # std = 0.
    classes_freq = np.zeros(8142)
    for _, y in tqdm(train_loader):
        # batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        # images = images.view(batch_samples, images.size(1), -1)
        # mean += images.mean(2).sum(0)
        # std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
    # mean /= len(train_loader.dataset)
    # std /= len(train_loader.dataset)
    print(classes_freq)
    # print(mean, std)