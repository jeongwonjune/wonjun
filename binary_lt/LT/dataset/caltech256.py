import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class ImbalanceCaltech256(VisionDataset):
    cls_num = 102

    def __init__(self, root: str, target_type="category", transform=None, target_transform=None, download: bool = False,
                 imb_type='exp', imb_factor=0.1, rand_number=0):
        super(ImbalanceCaltech256, self).__init__(os.path.join(root, 'caltech256'), transform=transform,
                                                  target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")
        self.categories = np.array(self.categories)

        classes_freq = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            classes_freq.append(n)
        class_idx = np.argsort(classes_freq)[::-1]
        self.categories = self.categories[class_idx]
        self.index = []
        self.targets = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.targets.extend(n * [i])

        self.samples = []
        for (i, c) in enumerate(self.categories):
            for file_name in os.listdir(os.path.join(self.root, "101_ObjectCategories", c)):
                self.samples.append(os.path.join(self.root, "101_ObjectCategories", c, file_name))
        print(len(self.targets), len(self.samples))
        self.samples = np.array(self.samples)
        self.index = np.array(self.index)
        self.targets = np.array(self.targets)
        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)
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
        self.data = np.array(new_data)
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img = Image.open(self.data[index])

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.targets[index])
        target = torch.tensor(target[0]).long()

        if self.transform is not None:
            img = self.transform(img)
        if img.shape == torch.Size([1, 224, 224]):
            img = img.repeat(3,1,1)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.data)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )

    def extra_repr(self):
        return "Target type: {target_type}".format(**self.__dict__)


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = ImbalanceCaltech256('/data', download=True, transform=train_transform)
    # test_dataset = ImbalanceCaltech256('/data', train=False, download=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    for images, y in train_loader:
        print(y)
    print(train_dataset.get_cls_num_list())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)
    mean = 0.
    std = 0.
    classes_freq = np.zeros(len(train_dataset.get_cls_num_list()))
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
