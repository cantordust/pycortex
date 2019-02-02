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
import torch.nn as tn
import torch.nn.functional as tnf
from torch.distributions import Categorical

import gym
import cv2
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.animation as anim
from collections import defaultdict

ROM = 'Pong-v0'

Actions = {
           'Pong-v0' : {0:0, 1:2, 2:5},
           'Breakout-v0' : {0:0, 1:3, 2:4}
           }

def preprocess(_state1, _state2):
    '''
    Crop, resize and return a grayscale version of the frame.
    '''
#    state = cv2.cvtColor(cv2.resize(_state, (84, 110)), cv2.COLOR_BGR2GRAY)[16:100,:]
    state = np.maximum(cv2.cvtColor(cv2.resize(_state1, (84, 110)), cv2.COLOR_BGR2GRAY)[16:100,:],
                       cv2.cvtColor(cv2.resize(_state1, (84, 110)), cv2.COLOR_BGR2GRAY)[16:100,:])

    return torch.from_numpy(np.reshape(state, (1,84,84))).float() / 255

def update_buffer(_env, _buffer, _action):
    '''
    Take a step and append it to the state buffer.
    '''

    ###################################################
    # Skip 2, take the max of the next two, visual mode
    ###################################################
#    step = 0
#    state1 = None
#    state2 = None
#    total_reward = 0.0

#    for it in range(4):
#        state, reward, done, _ = _env.step(_action)
#        total_reward += reward

#        if done:
#            break

#        step += 1

#        if step == 3:
#            state1 = state

#        if step == 4:
#            state2 = state

#        if (state1 is not None and
#            state2 is not None):
#            _buffer.append(preprocess(state1, state2))

#    return total_reward, done


    ##################################
    # Skip 2, take the third, RAM mode
    ##################################

#    total_reward = 0.0

#    for it in range(2):
#        state, reward, done, _ = _env.step(_action)
#        total_reward += reward

#        if done:
#            break

#    _buffer.append(torch.from_numpy(_env.unwrapped._get_ram()).float() / 255)

#    return total_reward, done

    ###############################
    # One state at a time, RAM mode
    ###############################

    state, reward, done, _ = _env.step(_action)
    _buffer.append(torch.from_numpy(_env.unwrapped._get_ram()).float() / 255)

    return reward, done

def init_buffer(_env, _buffer_size):

    buffer = ctx.Cont.Ring(_buffer_size)

    while len(buffer) < _buffer_size:

        # Update state buffer
        reward, done = update_buffer(_env, buffer, 0)
        if done:
            break

#    for n in range(len(buffer)):
#        mpl.matshow(buffer.data[n].data[0].numpy(), cmap='gray')
#        mpl.show()
#        ctx.pause()

    return buffer, done

def select_action(_net, _input):
    output = tnf.log_softmax(_net(_input), dim = 1)
#    print(f'Output: {output}')

    ###############################################
    # Choose an action from a weighted distribution
    ###############################################
    action_dist = Categorical(torch.exp(output))
#    action_dist = Categorical(-1 / output)
    action = action_dist.sample()

    ###############################
    # Always choose a greedy aciton
    ###############################
#    action = torch.argmax(output)

    return action.item(), output

def optimise(_net, _conf, _history, _optimiser, _lr_scheduler):

    # Zero out the optimiser gradients
    def closure():

        _optimiser.zero_grad()
        discounted_reward = 0

        raw_rewards = np.array(_history['reward'])
#        print(f'Raw rewards: {raw_rewards}')

        scaled_rewards = torch.zeros(len(raw_rewards))

        factor = 1.0 - _conf.discount_factor
        for idx, reward in reversed(list(enumerate(raw_rewards))):
            discounted_reward = reward + factor * discounted_reward
            scaled_rewards[idx] = discounted_reward

        mean = scaled_rewards.mean()
        sd = scaled_rewards.std()

        scaled_rewards = (scaled_rewards - mean) / (sd + _conf.epsilon)
#        print(f'Scaled rewards: {scaled_rewards}')

#        baseline = mean / sd
        baseline = 0

#        print(f'Normalised rewards: {scaled_rewards}')

        mask = torch.zeros_like(_history['output'])

        for idx, val in enumerate(_history['action']):
            mask[idx][val] = scaled_rewards[idx].item() - baseline

#        print(f'Mask: {mask}')

        losses = -torch.mul(mask, _history['output'])
#        print(f'Losses: {losses}')

        loss = (torch.sum(losses, 1)).mean()
#        print(f'Loss: {loss}')

        loss.backward()

        return loss

    if _lr_scheduler is not None:
        _lr_scheduler.step()

    _net.optimise(closure, _optimiser)

#    for param in _net.parameters():
#        print(param.grad)

def run_episode(_net,
                _conf,
                _env,
                _optimiser = None,
                _lr_scheduler = None,
                _train = False,
                _render = False,
                _animate = False):

    state = _env.reset()
    done = False
    buffer, done = init_buffer(_env, _conf.buffer_size)

    steps = 0
    total_reward = 0.0

    if _animate:
        frames = []

    if _train:

        history = {
                  'output': torch.zeros(0, *ctx.cn.Net.Output.Shape),
                  'action': [],
                  'reward': []
                  }

    while not done:
        action, output = select_action(_net, torch.cat(buffer.dump()).unsqueeze(0))
#        reward, done = update_buffer(_env, buffer, Actions[ROM][action])
        reward, done = update_buffer(_env, buffer, action)
        total_reward += reward

        if _train:
            history['action'].append(action)
            history['output'] = torch.cat((history['output'], output))
            history['reward'].append(reward)

        if _render:
            _env.render()
            time.sleep(0.02)

        if _animate:
            frame = mpl.imshow(buffer.data[buffer.head].data[0].numpy(), cmap='gray', animated=True)
            frames.append([frame])

        steps += 1

    if _train:
        optimise(_net, _conf, history, _optimiser, _lr_scheduler)
        _net.reset_recurrent_layers()

        history['output'] = torch.zeros(0, *ctx.cn.Net.Output.Shape)
        history['action'] = []
        history['reward'] = []

    if _animate:
        fig = mpl.figure()
        ani = anim.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        mpl.show()

    return steps, total_reward

def train(_net, _env, _conf):

    net = _net.to(_conf.device)

    # Train
    score = ctx.Stat.SMAStat()

    optimiser = _conf.optimiser(net.parameters(), **_conf.optimiser_args)
#    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.99)
    lr_scheduler = None

    for episode in range(1, _conf.episodes + 1):

        steps, total_reward = run_episode(net, _conf, _env, optimiser, lr_scheduler, _train = True)

        # Update the running score
        score.update(steps)

        print(f'[Episode {episode}] Steps: {steps:5d}\tMean loss: {net.fitness.loss_stat.mean:.3f}\tTotal reward: {total_reward:.0f}\tMean score: {score.mean:.2f}')

    # Render an episode
    run_episode(net, _conf, _env, _render = True)

    return net

def main():

    if ctx.get_rank() == 0:

        # This is the master process.
        # Parse command line arguments and set the default parameters.
        ctx.init_conf()

        # Temporary environment to get the input dimensions and other parameters
        env = gym.make(ROM)
        state1 = env.reset()
#        state2, _, _, _ = env.step(0)

        buffer_size = 4
        # Set the initial parameters
#        cn.Net.Input.Shape = [buffer_size, *list(preprocess(state1, state2).size())[1:]]
        cn.Net.Input.Shape = [buffer_size * 128]
#        cn.Net.Output.Shape = [len(Actions[ROM])]
        cn.Net.Output.Shape = [env.action_space.n]

#        cn.Net.Init.Layers = [ctx.cl.Layer.Def([10,3,3], [2,2])]
        cn.Net.Init.Layers = [ctx.cl.Layer.Def([64])]
        ctx.Conf.OptimiserArgs['lr'] = 0.1
        ctx.Conf.DiscountFactor = 0.01
        ctx.Conf.Epsilon = np.finfo(np.float32).eps.item()
        ctx.Conf.Episodes = 200

        # Allow recurrence for FC layers
#        cl.Layer.RecurrentFC = True

        ctx.print_conf()

        conf = ctx.Conf(0, 0)
        conf.buffer_size = buffer_size
        net = cn.Net()

        train(net, env, conf)

#        ctx.init()

#    # Run Cortex
#    ctx.execute()

if __name__ == '__main__':
    main()
