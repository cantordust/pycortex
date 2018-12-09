#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:52:44 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""

import torch
import torch.nn.functional as tnf
import torch.optim as optim
from torchvision import datasets, transforms

def train(net, device, train_loader, epoch):

    net.print()
    model = net.to(device)
    model.train()
    optimizer = optim.Adadelta(model.parameters())

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = tnf.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
#            if batch_idx % (500 * epoch)  == 0:
#                if (ctx.mRand.chance(0.9)):
#                    net.resize_kernel(ctx.mRand.uint(0,2))
#                else:
#                    if (ctx.mRand.chance(0.9)):
#                        net.add_nodes(ctx.mRand.uint(0,2))
#                    else:
#                        net.erase_nodes(ctx.mRand.uint(0,2))

#                net.resize_kernel(ctx.mRand.uint(0,2))
#                ctx.pause()

#            model = net.to(device)
#            optimizer = optim.Adadelta(model.parameters())
#            model.train()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return model, loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += tnf.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_loader(_batch_size, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = _batch_size, shuffle = True, **kwargs)
    return train_loader

def test_loader(_batch_size, **kwargs):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = _batch_size, shuffle = True, **kwargs)
    return test_loader
