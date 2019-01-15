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

import math

import torch
from torchvision import datasets, transforms

def get_loader(_conf, _train, _fitness = 0.0):

#    print('Data dir: {}'.format(_conf.data_dir))

    dataset = datasets.MNIST(_conf.data_dir,
                             train = _train,
                             transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    indices = list(torch.randperm(len(dataset)))

    if _train:
        indices = indices[0:math.floor(_fitness * len(dataset))]

    sampler = torch.utils.data.SubsetRandomSampler(indices)

    batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                  batch_size = _conf.train_batch_size,
                                                  drop_last = False)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_sampler = batch_sampler,
                                         **_conf.data_load_args)

    return loader

def test(_conf, _net):

    _net.eval()
    test_loss = 0
    correct = 0

    loader = get_loader(_conf, False)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(_conf.device), target.to(_conf.device)
            output = _conf.output_function(_net(data), **_conf.output_function_args)
            test_loss += _conf.loss_function(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    accuracy = 100. * correct / len(loader.dataset)
    print(f'[Net {_net.ID}] Test | Run {_conf.run} | ' +
          f'Epoch {_conf.epoch} Average loss: {test_loss:.4f}, ' +
          f'Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)')

    return accuracy

def train(_conf, _net):

    net = _net.to(_conf.device)
    net.train()
    optimiser = _conf.optimiser(net.parameters(), lr = 1.0 - net.fitness.relative)

    loader = get_loader(_conf, True, net.fitness.relative)

    net.fitness.loss_stat.reset()

    examples = 0
    for batch_idx, (data, target) in enumerate(loader):

        examples = batch_idx * len(data)
        data, target = data.to(_conf.device), target.to(_conf.device)
        net.optimise(data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)

    print(f'[Net {net.ID}] Test | Run {_conf.run} | Epoch {_conf.epoch} Trained on {100. * examples / len(loader.dataset):.2f}% of the dataset')
    net.fitness.set(test(_conf, net))

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
                           download=True)

        # Assign the train function
        ctx.Conf.Evaluator = train

        ctx.print_conf()

#        ctx.init()

    # Run Cortex
    ctx.run()

if __name__ == '__main__':
    main()
