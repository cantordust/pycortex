#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:52:44 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""

import cortex.cortex as ctx
import cortex.network as cn
import cortex.layer as cl

import torch
from torchvision import datasets, transforms

def get_train_loader(_conf):

#    print('Data dir: {}'.format(_conf.data_dir))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(_conf.data_dir,
                       train=True,
                       transform = transforms.ToTensor()),
                       batch_size = _conf.train_batch_size,
                       shuffle = True,
                       **_conf.data_load_args)

    return train_loader

def get_test_loader(_conf):

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(_conf.data_dir,
                       train=False,
                       transform = transforms.ToTensor()),
                       batch_size = _conf.test_batch_size,
                       shuffle = True,
                       **_conf.data_load_args)

    return test_loader

def test(_conf, _net):

    _net.eval()
    test_loss = 0
    correct = 0

    test_loader = get_test_loader(_conf)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(_conf.device), target.to(_conf.device)
            output = _conf.output_function(_net(data), **_conf.output_function_args)
            test_loss += _conf.loss_function(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\t[Net {} | Test | Epoch {}] Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        _net.ID, _conf.epoch, test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

def train(_conf, _net):

    _net = _net.to(_conf.device)
    _net.train()
    optimiser = _conf.optimiser(_net.parameters(), lr = 1.0 - _net.fitness.relative)

    train_loader = get_train_loader(_conf)

    _net.fitness.loss_stat.reset()

    for batch_idx, (data, target) in enumerate(train_loader):

        progress = batch_idx / len(train_loader)

        # Skip this training batch with probability proportional to the fitness and
        # inversely proportional to the epoch
        if ctx.Rand.chance(progress / _conf.epoch):
            continue

        data, target = data.to(_conf.device), target.to(_conf.device)

        _net.optimise(data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)

        if (batch_idx + 1) % _conf.log_interval == 0:
            print('[Net {} | Train | Epoch {}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                _net.ID, _conf.epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * progress, _net.fitness.loss_stat.current_value))

    _net.fitness.absolute = test(_conf, _net)
    _net.fitness.stat.update(_net.fitness.absolute)

    return _net

def main():

    if ctx.get_rank() == 0:

        # This is the master process.
        # Parse command line arguments and set default parameters
        ctx.init_conf()

        # Set the initial parameters
        cn.Net.Input.Shape = [1, 28, 28]
        cn.Net.Output.Shape = [10]
        cn.Net.Init.Layers = []

        # If necessary, run the train loader to download the data
        if ctx.Conf.DownloadData:
            datasets.MNIST(ctx.Conf.DataDir,
                           download=True,
                           transform=transforms.Normalize((0.1307,), (0.3081,)))

        # Assign the train function
        ctx.Conf.Evaluator = train

    # Run Cortex
#    ctx.init()
    ctx.run()

if __name__ == '__main__':
    main()
