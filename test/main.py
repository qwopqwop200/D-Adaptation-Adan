'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np
import random

from models import *
from opt import *
from utils import progress_bar

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch,args):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'opt': args.opt,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.opt}_ckpt.pth')
        best_acc = acc
    return acc

if __name__ == '__main__':
    fix_seed(42)
    acc_log = []

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--opt', default='adam', type=str, choices=['adam','adan','d-adam','d-adam-ip','d-adan','d-adan-ip'], help='optimizer')
    parser.add_argument('--restart', action='store_true', help='restart strategy in adan' )
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')

    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    elif args.opt == 'adan':
        optimizer = Adan(net.parameters(), lr=1e-3)
    elif args.opt == 'd-adam':
        optimizer = DAdaptAdam(net.parameters(), lr=1.0)
    elif args.opt == 'd-adam-ip':
        optimizer = DAdaptAdamIP(net.parameters(), lr=1.0)
    elif args.opt == 'd-adan':
        optimizer = DAdaptAdan(net.parameters(), lr=1.0)
    elif args.opt == 'd-adan-ip':  
        optimizer = DAdaptAdanIP(net.parameters(), lr=1.0)
    else:
        raise NotImplementedError
      
    if args.restart and 'adan' in args.opt: 
        args.restart = True
        args.opt += '_restart'
    else:
        args.restart = False
      
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(start_epoch, 50):
        train(epoch)
        acc = test(epoch,args)
        acc_log.append(acc)
        if args.restart and (epoch+1) % 10 == 0:
            print('restart')
            optimizer.restart_opt()
        scheduler.step()
    print(f'opt:{args.opt},best_acc:{best_acc}')
    
    with open(f'./checkpoint/{args.opt}_log.txt', 'w') as f:
        for acc in acc_log:
            data = f'{acc}\n'
            f.write(data)
        f.write(f'best_acc:{best_acc}')
