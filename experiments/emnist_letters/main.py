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

def get_loader(_dir,
               _batch_size = 1,
               _train = True,
               _portion = 1.0,
               _download = False,
               **_args):

    dataset = datasets.EMNIST(_dir,
                             split = 'letters',
                             train = _train,
                             download = _download,
                             transform = transforms.Compose([
                                transforms.ToTensor()
                             ]))

    indices = list(torch.randperm(len(dataset)))

    if _train:
        indices = indices[0:math.floor(_portion * len(dataset))]

    sampler = torch.utils.data.SubsetRandomSampler(indices)

    batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                  batch_size = _batch_size,
                                                  drop_last = False)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_sampler = batch_sampler,
                                         **_args)

    return loader

def test(_conf, _net):

    _net.eval()
    test_loss = 0
    correct = 0

    loader = _conf.data_loader(_dir = _conf.data_dir,
                               _batch_size = _conf.test_batch_size,
                               _train = False,
                               _download = _conf.download_data,
                               **_conf.data_load_args)

    with torch.no_grad():
        for data, target in loader:
            target -= 1
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

    net = _net.to(_conf.device, non_blocking=True)
    net.train()
    optimiser = _conf.optimiser(net.parameters())

    loader = _conf.data_loader(_dir = _conf.data_dir,
                               _batch_size = _conf.train_batch_size,
                               _train = True,
                               _portion = net.complexity if _conf.train_portion is None else _conf.train_portion,
                               **_conf.data_load_args)

    net.fitness.loss_stat.reset()

    examples = 0
    for batch_idx, (data, target) in enumerate(loader):
        target -= 1
        examples += len(data)
        data, target = data.to(_conf.device), target.to(_conf.device)
        net.optimise(data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)

    print(f'[Net {net.ID}] Train | Run {_conf.run} | Epoch {_conf.epoch} Trained on {100. * examples / len(loader.dataset):.2f}% of the dataset')
    net.fitness.set(test(_conf, net))

    return net

def main():

    if ctx.get_rank() == 0:

        # This is the master process.
        # Parse command line arguments and set default parameters
        ctx.init_conf()

        # Set the initial parameters
        cn.Net.Input.Shape = [1, 28, 28]
        cn.Net.Output.Shape = [26]
        cn.Net.Init.Layers = []

        ctx.Conf.DataLoader = get_loader
        ctx.Conf.Evaluator = train

        ctx.print_conf()

        # Download the data if necessary
        if ctx.Conf.DownloadData:

            try:
                loader = ctx.Conf.DataLoader(_dir = ctx.Conf.DataDir,
                                             _download = True)

                ctx.Conf.DownloadData = False

            except:
                print('Error downloading data')
                ctx.dump_exception()
                ctx.Conf.Tag = ctx.Tags.Exit

    # Run Cortex
    ctx.run()

if __name__ == '__main__':
    main()
