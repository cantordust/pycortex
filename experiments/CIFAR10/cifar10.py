#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:52:44 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""

import torch.multiprocessing as tm
import torch
from torchvision import datasets, transforms

import cortex.cortex as ctx
import cortex.network as cn
import cortex.layer as cl

loader_lock = tm.Lock()

def get_train_loader(_conf):

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(_conf.data_dir,
                         train=True,
                         transform = transforms.ToTensor()),
                         batch_size = _conf.train_batch_size,
                         shuffle = True,
                         **_conf.data_load_args)

    return train_loader

def get_test_loader(_conf):

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(_conf.data_dir,
                         train=False,
                         transform = transforms.ToTensor()),
                         batch_size = _conf.test_batch_size,
                         shuffle = True,
                         **_conf.data_load_args)

    return test_loader

def test(_net, _conf):

    _net.eval()
    test_loss = 0
    correct = 0

    with loader_lock:
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
    print('\n[Net {}] Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        _net.ID, test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

def train(_net, _epoch, _conf):

    _net = _net.to(_conf.device)
    _net.train()
    optimiser = _conf.optimiser(_net.parameters())

    with loader_lock:
        train_loader = get_train_loader(_conf)

    _net.fitness.loss_stat.reset()
    train_portion = 1.0 - _net.fitness.relative

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(_conf.device), target.to(_conf.device)

        _net.optimise(data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)
        progress = batch_idx / len(train_loader)

        if (batch_idx + 1) % _conf.log_interval == 0:
            print('[Net {}] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                _net.ID, _epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * progress, _net.fitness.loss_stat.current_value))

        if progress >= train_portion:
            break

    _net.fitness.absolute = test(_net, _conf)

    return _net

def main():

    # Parse command line arguments and set default parameters
    ctx.init_conf()

    cn.Net.Input.Shape = [1, 32, 32]
    cn.Net.Output.Shape = [10]
    cn.Net.Init.Layers = []

    # If necessary, run the train loader to download the data
    if ctx.Conf.DownloadData:
        datasets.CIFAR10(ctx.Conf.DataDir,
                         download=True,
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    ctx.Conf.Evaluator = train

    # Print the current configuration
    ctx.print_conf()

    # Run Cortex
#    ctx.init()
    ctx.run()

if __name__ == '__main__':
    main()
