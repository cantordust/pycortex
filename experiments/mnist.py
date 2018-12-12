#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:52:44 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""

import torch
from torchvision import datasets, transforms
from torch.autograd import detect_anomaly

#import sys
#sys.path.append("..")

from cortex import cortex as ctx

def get_train_loader():

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = ctx.TrainBatchSize, shuffle = True, **ctx.DataLoadArgs)
    return train_loader

def get_test_loader():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = ctx.TestBatchSize, shuffle = True, **ctx.DataLoadArgs)
    return test_loader

def test(net):

    net.eval()
    test_loss = 0
    correct = 0

    test_loader = get_test_loader()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(ctx.Device), target.to(ctx.Device)
            output = net(data)
            test_loss += ctx.LossFunction(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train(net, epoch):

    net = net.to(ctx.Device)
    net.train()
    optimiser = ctx.Optimiser(net.parameters())

    train_loader = get_train_loader()

    for batch_idx, (data, target) in enumerate(train_loader):
#        with detect_anomaly():
        data, target = data.to(ctx.Device), target.to(ctx.Device)
        optimiser.zero_grad()
        output = net(data)
        loss = ctx.LossFunction(output, target)
        loss.backward()
        optimiser.step()
        if (batch_idx + 1) % ctx.LogInterval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#        if (batch_idx + 1) % (30 * ctx.LogInterval) == 0:
#            net.mutate()
##            net = net.to(ctx.Device)
##            net.train()
#            optimiser = ctx.Optimiser(net.parameters())

    return net
