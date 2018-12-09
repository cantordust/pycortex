#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:42:46 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""
import argparse

import torch

from .kernel import kernel as knl
from .experiments import mnist

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # Device, RNG seed, etc.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed()

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Data loaders
#    batch_size = ctx.mConf.Learning.BatchSize
    train_loader = mnist.train_loader(args.train_batch_size, **kwargs)
    test_loader = mnist.test_loader(args.test_batch_size, **kwargs)

    # Network
    p1 = ctx.Net()
    p1.add_layer(_shape = [5, 0, 0], _layer_index = 1)
    p2 = ctx.Net()
    p2.add_nodes(2, 2)

    p1.add_layer(_shape = [10], _layer_index = 3)

    # Train / test loop

    for ID, net in ctx.Net.population.items():
#        for epoch in range(1, args.epoch + 1):
        for epoch in range(1, 2):

            model, loss = mnist.train(net, device, train_loader, epoch)
            mnist.test(model, device, test_loader)

    offspring = ctx.Net(_p1 = ctx.Net.population[1], _p2 = ctx.Net.population[2])
    offspring.print()

#    for epoch in range(1, 2):
#        model, loss = mnist.train(offspring, device, train_loader, epoch)
        mnist.test(model, device, test_loader)

if __name__ == '__main__':
    main()
