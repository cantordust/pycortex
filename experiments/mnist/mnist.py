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
    print(f'[Net {_net.ID}] Test | Run {_conf.run} | ' +
          f'Epoch {_conf.epoch} Average loss: {test_loss:.4f}, ' +
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

    return accuracy

def train(_conf, _net):

    net = _net.to(_conf.device)
    net.train()
    optimiser = _conf.optimiser(net.parameters())

    train_loader = get_train_loader(_conf)

    net.fitness.loss_stat.reset()

    examples = 0
    for batch_idx, (data, target) in enumerate(train_loader):

#        progress = batch_idx * len(data) / len(train_loader.dataset)

        # Skip this training batch with probability determined by the network complexity
        if ctx.Rand.chance(1.0 - net.fitness.relative):
            continue

        examples += len(data)

        data, target = data.to(_conf.device), target.to(_conf.device)

        net.optimise(data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)

#        if (batch_idx + 1) % _conf.log_interval == 0:
#            print(f'[Net {net.ID}] Train | Run {_conf.run} | ' +
#                  f'Epoch {_conf.epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' +
#                  f'({100. * progress:.0f}%)] Loss: {net.fitness.loss_stat.current_value:.6f}')

    print(f'[Net {net.ID}] Test | Run {_conf.run} | Epoch {_conf.epoch} Trained on {100. * examples / len(train_loader.dataset):.2f}% of the dataset')
    net.fitness.absolute = test(_conf, net)
    net.fitness.stat.update(net.fitness.absolute)

    return net

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

        ctx.print_conf()

#        ctx.init()

    # Run Cortex
    ctx.run()

if __name__ == '__main__':
    main()
