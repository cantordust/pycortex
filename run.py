#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:42:46 2018

@author: Alexander Hadjiivanov
@license: MIT ((https://opensource.org/licence/MIT)
"""
import argparse
import torch

from cortex import cortex as ctx
from experiments.MNIST import mnist

def parse():

    # Training settings
    parser = argparse.ArgumentParser(description='PyCortex argument parser')

    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--runs', type=int, default=1, metavar='N',
                        help='number of runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--init-nets', type=int, default=32, metavar='N',
                        help='Initial number of networks (default: 32)')
    parser.add_argument('--max-nets', type=int, default=256, metavar='N',
                        help='Maximal number of networks (default: 256)')
    parser.add_argument('--init-species', type=int, default=8, metavar='N',
                        help='Initial number of species (default: 8)')
    parser.add_argument('--max-species', type=int, default=32, metavar='N',
                        help='Maximal number of species (default: 32)')
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
    parser.add_argument('--experiment-name', type=str, default='Experiment', metavar='S',
                        help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='./logs', metavar='N',
                        help='Directory for storing the output logs')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    if args.train_batch_size:
        ctx.TrainBatchSize = args.train_batch_size

    if args.test_batch_size:
        ctx.TestBatchSize = args.test_batch_size

    if args.runs:
        ctx.Runs = args.runs

    if args.epochs:
        ctx.Epochs = args.epochs

    if args.init_nets:
        ctx.Net.Init.Count = args.init_nets

    if args.max_nets:
        ctx.Net.Max.Count = args.max_nets

    if args.init_species:
        ctx.Species.Init.Count = args.init_species

    if args.max_species:
        ctx.Species.Max.Count = args.max_species

    if args.lr:
        ctx.LearningRate = args.lr

    if args.momentum:
        ctx.Momentum = args.momentum

    if args.cuda and torch.cuda.is_available():
        ctx.Device = torch.device('cuda')
        ctx.DataLoadArgs = {'num_workers': 1,
                            'pin_memory': True}

    if args.rand_seed is not None:
        torch.manual_seed(args.rand_seed)

    if args.max_threads is not None:
        ctx.MaxThreads = args.max_threads

    if args.experiment_name is not None:
        ctx.ExperimentName = args.experiment_name

    if args.log_dir:
        ctx.LogDir = args.log_dir

    if args.log_interval:
        ctx.LogInterval = args.log_interval

def main():

    ctx.Net.Init.Layers = [ctx.Layer.Def(10)]

    # Parse command line arguments
    parse()

    # Print the current configuration
    ctx.print_config()

    # Run Cortex
#    ctx.init()
    ctx.run()

if __name__ == '__main__':
    main()
