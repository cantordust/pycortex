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
from experiments import mnist

def parse():

    # Training settings
    parser = argparse.ArgumentParser(description='PyCortex argument parser')

    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--runs', type=int, default=1, metavar='N',
                        help='number of runs (default: 1)')
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

    if args.experiment_name is not None:
        ctx.ExperimentName = args.experiment_name

    if args.log_dir:
        ctx.LogDir = args.log_dir

    if args.log_interval:
        ctx.LogInterval = args.log_interval

def main():

    # Parse command line arguments
    parse()

    # Set any other options
    ctx.Net.Input.Shape = [1, 28, 28]
    ctx.Net.Output.Shape = [10]

    ctx.Net.Init.Layers = [ctx.Layer.Def(10)]

    ctx.Epochs = 20

    ctx.Net.Init.Count = 4
    ctx.Net.Max.Count = 32

    ctx.Species.Init.Count = 2
    ctx.Species.Max.Count = 8

    ctx.TrainFunction = mnist.train

    # Print the current configuration
    ctx.print_config()

    # Run Cortex
#    ctx.init()
    ctx.run()

if __name__ == '__main__':
    main()
