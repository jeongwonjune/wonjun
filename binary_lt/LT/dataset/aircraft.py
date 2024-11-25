import random

import numpy as np
import os

import torch
import torchvision
import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive


class Aircraft(VisionDataset):
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None, rand_number=0,
                 target_transform=None, download=False, imb_factor=1, imb_type='exp'):
        np.random.seed(rand_number)
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        self.targets = targets
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
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

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        target = torch.tensor(int(target)).long()
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
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


if __name__ == '__main__':

    # train_dataset = Aircraft(root='/data', train=True, download=False, transform=train_transform, imb_factor=1)
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
    #     plt.savefig(f'Aircraft_{i}.png')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = Aircraft('/data', train=True, download=True, transform=train_transform, imb_factor=0.1)
    test_dataset = Aircraft('/data', train=False, download=True, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # classes_freq = np.zeros(100)
    # for x, y in tqdm.tqdm(train_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    #
    # for x, y in tqdm.tqdm(test_loader):
    #
    #

    # print(train_dataset.get_cls_num_list())
    mean = 0.
    std = 0.
    classes_freq = np.zeros(100)
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