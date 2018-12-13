#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:42:46 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""
import argparse
import torch

from tensorboardX import SummaryWriter

from cortex import cortex as ctx
from experiments import mnist

def parse():

    # Training settings
    parser = argparse.ArgumentParser(description='PyCortex argument parser')

    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--runs', type=int, default=100, metavar='N',
                        help='number of runs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--rand-seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--max-threads', type=int, default=None, metavar='S',
                        help='number of threads (default: all available cores)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    if args.train_batch_size:
        ctx.Net.TrainBatchSize = args.train_batch_size

    if args.test_batch_size:
        ctx.Net.TestBatchSize = args.test_batch_size

    if args.epochs:
        ctx.Net.Epochs = args.epochs

    if args.runs:
        ctx.Net.Runs = args.runs

    if args.lr:
        ctx.Net.LearningRate = args.lr

    if args.momentum:
        ctx.Net.Momentum = args.momentum

    if args.cuda and torch.cuda.is_available():
        ctx.Net.Device = torch.device('cuda')
        ctx.Net.DataLoadArgs = {'num_workers': 1,
                            'pin_memory': True}

    if args.rand_seed is not None:
        torch.manual_seed(args.rand_seed)

    if args.max_threads is not None:
        ctx.MaxThreads = args.max_threads

    if args.log_interval:
        ctx.Net.LogInterval = args.log_interval

def main():

    # Parse command line arguments
    parse()

    # Set any other options
    #ctx.Net.Input.Shape = [1, 28, 28]
    #ctx.Net.Output.Shape = [10]
    ctx.Epochs = 1
    ctx.LogInterval = 50
    ctx.Net.Init.Count = 2
    ctx.Species.Init.Count = 1
    ctx.Species.Max.Count = 2

    # Print the current configuration
    ctx.print_config()

    # Initialise Cortex
    ctx.init()

    for net in ctx.Net.ecosystem.values():
#        for i in range(5):
#            net.mutate(_parameters = False)

        for i in range(20):
            net.mutate(_structure = False)

    print("Species:", len(ctx.Species.populations))
    print("Nets:", len(ctx.Net.ecosystem))
    for species in ctx.Species.populations.values():
        print("Species", species.ID, "contains networks", *sorted(species.nets))

    net1 = ctx.Net.ecosystem[1]
    net2 = ctx.Net.ecosystem[2]

    net1.print()
    net2.print()
    offspring = ctx.Net(_p1 = net1, _p2 = net2)

    offspring.print()

    for epoch in range(1, ctx.Epochs + 1):
        net1 = mnist.train(net1, epoch)
        mnist.test(net1)

    for epoch in range(1, ctx.Epochs + 1):
        net2 = mnist.train(net2, epoch)
        mnist.test(net2)

    mnist.test(offspring)
#
#    for epoch in range(1, ctx.Epochs + 1):
#        offspring = mnist.train(offspring, epoch)
#        mnist.test(offspring)

#    dummy_input = torch.randn(1, *ctx.Net.Input.Shape)
#    with SummaryWriter(comment='Cortex network') as w:
#        w.add_graph(model, dummy_input, True)

if __name__ == '__main__':
    main()
