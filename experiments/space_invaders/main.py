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
import time

import math

import torch
import gym
import cv2
import numpy as np
import matplotlib as mpl
from collections import namedtuple
import torch.sparse as tsp

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)[26:110,:]
    return np.reshape(observation,(84,84,1))

def train(_conf, _net):

    env = gym.make('SpaceInvaders-v0')
    env.reset()
    env.render()

    observation = env.reset()

    net = _net.to(_conf.device)

    # Train the network if it is not a new offspring
    if net.age > 0:
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

            examples += len(data)
            data, target = data.to(_conf.device), target.to(_conf.device)
            net.optimise(data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)

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

#        ctx.init()

    # Run Cortex
    ctx.execute()

if __name__ == '__main__':
#    main()
    env = gym.make('SpaceInvaders-v0')
    observation = env.reset()
    done = False

    actions = [n for n in range(env.action_space.n)]
    weights = [1.0] * len(actions)
    lr = 0.0001

    state_buffer = ctx.Cont.Ring(3)

    state_buffer.push(env.reset())
    state_buffer.push(env.step(0))
    state_diff = state_buffer.buffer[0] - state_buffer.buffer[0]
    state_buffer.push(state_diff)

    while not done:
#    for _ in range(300):

        action = np.random.choice(actions, p = ctx.Func.softmax(weights))
        observation, reward, done, info = env.step(action)
        print(info)
        print(env.action_space)
        print(action)

        tensor = (torch.from_numpy(preprocess(observation)).float() / 255).half()
        for idx, act in enumerate(actions):
            if reward > 0:
                if act == action:
                    weights[idx] *= (1.0 + reward * lr)
                else:
                    weights[idx] *= (1.0 - reward * lr) / (len(actions) - 1)
            else:
                if act == action:
                    weights[idx] *= (1.0 - reward * lr)
                else:
                    weights[idx] *= (1.0 + reward * lr) / (len(actions) - 1)

        sparse_tensor = tsp.FloatTensor(tensor.size())
        sparse_tensor = 1
        print(sparse_tensor)

#        print(tensor)
#        done = True

        time.sleep(0.01)
        env.render()
