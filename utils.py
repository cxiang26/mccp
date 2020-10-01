
import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args, use_gpu=True):
    if args.net == 'resnet18':
        from resnet import resnet18
        net = resnet18(args)
    elif args.net == 'resnet34':
        from resnet import resnet34
        net = resnet34(args)
    elif args.net == 'resnet50':
        from resnet import resnet50
        net = resnet50(args)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset='cifar10'):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if dataset == 'cifar10':
        cifar10_training = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True,
                                                          transform=transform_train)
        cifar10_training_loader = DataLoader(
            cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
        return cifar10_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset='cifar10'):
    transform_test = transforms.Compose([
        # transforms.RandomRotation(75),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if dataset == 'cifar10':
        cifar10_test = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True,
                                                      transform=transform_test)
        cifar10_test_loader = DataLoader(
            cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

        return cifar10_test_loader

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
