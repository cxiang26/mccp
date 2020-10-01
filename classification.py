# train.py
#!/usr/bin/env	python3

import os
import sys
import argparse
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

cudnn.benchmark = True

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
        print('Train/loss', loss.item(), n_iter)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('[epoch: {}] Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))

    return correct.float() / len(test_loader.dataset)

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-Ddim', type=int, default=4, help='the dimension of capsule subspace')
    parser.add_argument('-type', type=str, default='mccp', help='methods of classification, choosing from [fc mccp]')
    parser.add_argument('-savepath', type=str, default='./ablationstudy', help='save path')
    parser.add_argument('-dataset', type=str, default='cifar10')
    args = parser.parse_args()

    #data preprocessing:
    if args.dataset == 'cifar10':
        training_loader = get_training_dataloader(
            settings.CIFAR10_TRAIN_MEAN,
            settings.CIFAR10_TRAIN_STD,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s,
            dataset=args.dataset
        )

        test_loader = get_test_dataloader(
            settings.CIFAR10_TRAIN_MEAN,
            settings.CIFAR10_TRAIN_STD,
            num_workers=args.w,
            batch_size=args.b,
            shuffle=args.s,
            dataset=args.dataset
        )
        num_classes = 10

    args.num_classes = num_classes
    net = get_network(args, use_gpu=args.gpu)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.savepath, args.dataset, args.net, args.type)

    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}-{acc}.pth')
    print(checkpoint_path)
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        end = time.time()
        train(epoch)
        acc = eval_training(epoch)

        epoch_time = time.time() - end
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time * (settings.EPOCH - epoch))
        print('[epoch: {}]  [Need time: {:02d}:{:02d}:{:02d}]'.format(epoch, need_hour, need_mins, need_secs))
        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best', acc=acc))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular', acc=acc))
