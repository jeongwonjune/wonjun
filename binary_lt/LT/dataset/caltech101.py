import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class ImbalanceCaltech101(VisionDataset):
    cls_num = 101
    def __init__(self, root: str, target_type="category", train=True, transform=None, target_transform=None,
                 download: bool = True,
                 imb_type='exp', imb_factor=0.1, rand_number=42):
        super(ImbalanceCaltech101, self).__init__(os.path.join(root, 'caltech101'), transform=transform,
                                                  target_transform=target_transform)
        np.random.seed(rand_number)
        os.makedirs(self.root, exist_ok=True)
        if download:
            self.download()
            
        if train:
            data_txt = open(os.path.join(self.root, 'caltech101_train.txt'), 'r')
        else:
            data_txt = open(os.path.join(self.root, 'caltech101_test.txt'), 'r')

        self.samples = []
        self.targets = []
        while True:
            line = data_txt.readline()
            if not line: break
            file_path = os.path.join(self.root, "101_ObjectCategories", line.split(' ')[0])
            label = int(line.split(' ')[1])
            self.samples.append(file_path)
            self.targets.append(label)
        # if not isinstance(target_type, list):
        #     target_type = [target_type]
        # self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        # self.categories.remove("BACKGROUND_Google")
        self.categories = np.array(self.categories)

        # print(len(self.targets), len(self.samples))

        # retain_idx = []
        # for idx, label in enumerate(self.targets):
        #     if label != 4:
        #         retain_idx.append(idx)
        #         if label > 4:
        #             self.targets[idx] -= 1

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)

        # self.samples = self.samples[retain_idx]
        # self.targets = self.targets[retain_idx]

        tmp_idx = np.argsort(self.targets)
        self.samples = self.samples[tmp_idx]
        self.targets = self.targets[tmp_idx]

        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.targets) / cls_num
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
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.extend(self.samples[selec_idx])
            new_targets.extend([the_class, ] * len(self.samples[selec_idx]))
        # new_data = np.vstack(new_data)
        self.samples = np.array(new_data)
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")

        target = []
        target.append(self.targets[index])
        target = torch.tensor(target[0]).long()

        if self.transform is not None:
            if isinstance(self.transform, list):
                if type(self.transform) == list:
                    samples = []
                    seed = np.random.randint(2147483647)
                    for transform in self.transform:
                        random.seed(seed)
                        torch.random.manual_seed(seed)
                        sample = transform(img)
                        samples.append(sample)
                    return samples, target
            else:
                img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.samples)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self):
        return "Target type: {target_type}".format(**self.__dict__)


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ImbalanceCaltech101('/data', download=True, train=True, transform=train_transform)
    test_dataset = ImbalanceCaltech101('/data', train=False, transform=train_transform)

    # train_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=128, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)
    # i =0
    # for images, y in train_loader:
    #     plt.figure(dpi=400)
    #     print(y)
    #     plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    #     plt.savefig(f'{i}.png')
    #     i+=1
    # print(train_dataset.get_cls_num_list())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)
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