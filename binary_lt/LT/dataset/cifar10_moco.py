import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets
import sys
sys.path.append('./LT')
from utils.randaugment import rand_augment_transform


rgb_mean = (0.4914, 0.4822, 0.4465)
ra_params = dict(translate_const=int(32 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
base_moco_aug = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, is_moco=False,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        if is_moco:
            self.moco_transform = [transforms.Compose(base_moco_aug), transforms.Compose(base_moco_aug)]
        print(self.get_cls_num_list())

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
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
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.moco_transform is not None:
            sample1 = self.moco_transform[0](img)
            sample2 = self.moco_transform[1](img)
            return [sample1, sample2], label
        else:
            if self.transform:
                img = self.transform(img)
            return img, label

class CIFAR10_LT(object):
    def __init__(self, root='./data/cifar10', imb_type='exp',
                 imb_factor=0.01, batch_size=128, num_works=8):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                          train=True, download=True, transform=train_transform)
        moco_dataset = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                          train=True, download=True, transform=train_transform, is_moco=True)
        eval_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)

        self.cls_num_list = train_dataset.get_cls_num_list()

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True)

        self.train_moco_loader = torch.utils.data.DataLoader(
            moco_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, drop_last=True,
        )

        self.val_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)


if __name__ == '__main__':
    dataset = CIFAR10_LT(num_works=0)
    moco_loader = dataset.train_moco_loader
    print(len(moco_loader))
    for i, (images, target) in enumerate(moco_loader):
        print(images[0].shape, images[1].shape, target.shape)
        break