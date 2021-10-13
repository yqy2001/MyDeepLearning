"""
@Author:        禹棋赢
@StartTime:     2021/8/4 18:51
@Filename:      cifar.py
"""
import torch
import torchvision
import torchvision.transforms as transforms
from easydict import EasyDict

import sys

if sys.platform == "win64" or sys.platform == "win32":
    _DATA_cifar10 = ".\data\cifar10"
    _DATA_cifar100 = "datasets/cifar100/"
else:
    _DATA_cifar10 = "/home/mixiaoyue/yqy/A_adversarial_repo/datasets/cifar10/"
    _DATA_cifar100 = "datasets/cifar100/"

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def load_cifar10(bs):
    """
    Load training and test data of cifar10
    each image is [3, 32, 32]
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=_DATA_cifar10, train=True, transform=train_transforms, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=_DATA_cifar10, train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)
