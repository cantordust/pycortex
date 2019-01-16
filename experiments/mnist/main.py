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

    dataset = datasets.MNIST(_dir,
                             train = _train,
                             download = _download,
                             transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
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

def main():

    if ctx.get_rank() == 0:

        # This is the master process.
        # Parse command line arguments and set default parameters
        ctx.init_conf()

        # Set the initial parameters
        cn.Net.Input.Shape = [1, 28, 28]
        cn.Net.Output.Shape = [10]
        cn.Net.Init.Layers = []

        ctx.Conf.DataLoader = get_loader

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

        ctx.init()

    # Run Cortex
#    ctx.run()

if __name__ == '__main__':
    main()
