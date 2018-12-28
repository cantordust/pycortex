#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:52:44 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""

import torch
from torchvision import datasets, transforms

import cortex.cortex as ctx

loader_lock = torch.multiprocessing.Lock()

def get_train_loader(_data_dir):

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(_data_dir,
                       train=True,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       batch_size = ctx.TrainBatchSize,
                       shuffle = True,
                       **ctx.DataLoadArgs)

    return train_loader

def get_test_loader(_data_dir):

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(ctx.DataDir,
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       batch_size = ctx.TestBatchSize,
                       shuffle = True,
                       **ctx.DataLoadArgs)

    return test_loader

def test(net, _data_dir):

    net.eval()
    test_loss = 0
    correct = 0

    test_loader = get_test_loader(_data_dir)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(ctx.Device), target.to(ctx.Device)
            output = net(data)
            test_loss += ctx.LossFunction(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\n[Net {}] Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        net.ID, test_loss, correct, len(test_loader.dataset),
        accuracy))

    net.fitness.absolute = accuracy

def train(net, epoch, ecosystem, _data_dir):

    net = net.to(ctx.Device)
    net.train()
    optimiser = ctx.Optimiser(net.parameters())

    train_loader = get_train_loader(_data_dir)

    net.fitness.loss_stat.reset()
    train_portion = 1.0 - net.fitness.relative

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(ctx.Device), target.to(ctx.Device)

        net.optimise(data, target, optimiser)
        progress = batch_idx / len(train_loader)

        if (batch_idx + 1) % ctx.LogInterval == 0:
            print('[Net {}] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                net.ID, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * progress, net.fitness.loss_stat.current_value))

        if progress >= train_portion:
            break

    test(net, _data_dir)

    ecosystem[net.ID] = net
#    return net

def main():

    ctx.Net.Input.Shape = [1, 28, 28]
    ctx.Net.Output.Shape = [10]
    ctx.TrainFunction = train

    ctx.Net.Init.Layers = []

    # Parse command line arguments
    ctx.parse()

    # Print the current configuration
    ctx.print_config()

    # If necessary, run the train loader to download the data
    if ctx.DownloadData:
        datasets.MNIST(ctx.DataDir, download=True)

    # Run Cortex
#    ctx.init()
    ctx.run()

if __name__ == '__main__':
    main()
