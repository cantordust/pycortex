#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:33:58 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import random
random.seed()

import math
import copy
from colorama import Fore, Style

import torch
import torch.nn as tn

torch.set_printoptions(precision = 4, threshold = 5000, edgeitems = 5, linewidth = 160)

from cortex import cortex as ctx
from cortex.layer import Layer

import cortex.functions as Func
import cortex.statistics as Stat
import cortex.rnd as Rand
from cortex.rnd import RouletteWheel

def pass_fail(cond, *args):

    print(f'[ {Fore.GREEN}Passed{Style.RESET_ALL} ]' if cond else f'[ {Fore.RED}Failed{Style.RESET_ALL} ]', *args)

    return cond

def test_fmap(_val = 1.0):

    for enum, f in Func.fmap.items():
        print(enum.name, f(_val))

def test_stat(_type = Stat.MAType.Simple):

    stat = Stat.SMAStat("Test stats") if _type == Stat.MAType.Simple else Stat.EMAStat()
    for i in range(1, 100):
        stat.update(i)

    stat.print()

def test_rand():

    print("ND:", Rand.ND())
    print("negND:", Rand.negND())
    print("posND:", Rand.posND())
    print("ureal(-100.0,100.0):", Rand.ureal(-100.0, 100.0))
    print("uint(-100,100):", Rand.uint(-100, 100))
    array = [1,2,3,4,5]
    weights = [5, 3, 100, 67, 22]
    table = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    print("elem:", Rand.elem(array))
    print("roulette:", Rand.roulette(array, weights))
    print("key:", Rand.key(table))
    print("val:", Rand.val(table))

def test_torch_allclose():

    tensor1 = torch.ones(3,3)
    tensor2 = torch.ones(3,3)
    tensor2[0][0] += 1e-8

    print("tensor1:\n", tensor1)
    print("tensor2:\n", tensor2)

    print("tensor1 == tensor2", torch.allclose(tensor1, tensor2, 1e-8, 1e-8))

def test_print_module_params():

    layer = tn.Conv2d(2,3,3, bias = False)

    print(layer)

    print("layer parameters:\n", layer)
    for param in layer.parameters():
        print(param)

def test_init_tensor(_tensor = torch.zeros(3,3,3),
                     _func = tn.init.constant_,
                     _args = {'val': 4}):

    print("Original tensor:\n", _tensor)

    _func(_tensor, **_args)

    print("\nInitialised tensor:\n", _tensor)

def test_output_shape(_input_shape = [3, 32, 32],
                      _output_nodes = 10,
                      _kernel_size = (3, 3),
                      _padding = (),
                      _stride = (),
                      _dilation = ()):

    layer_types = {
        1: tn.Conv1d,
        2: tn.Conv2d,
        3: tn.Conv3d
        }

    if len(_kernel_size) == 0:
        layer = tn.Linear(_input_shape[0], _output_nodes)
    else:

        if len(_padding) == 0:
            _padding = tuple([dim // 2 for dim in _kernel_size])

        if len(_stride) == 0:
            _stride = tuple([1] * len(_kernel_size))

        if len(_dilation) == 0:
            _dilation = tuple([1] * len(_kernel_size))

        layer = layer_types[len(_kernel_size)](_input_shape[0], _output_nodes, _kernel_size, _stride, _padding, _dilation)

    layer_shape = list(layer.weight.size())

    del layer_shape[1] # Input channels

    print("Output nodes:", _output_nodes)
    print("Input shape:", _input_shape)
    print("Kernel size:", _kernel_size)
    print("Padding:", _padding)
    print("Striide:", _stride)
    print("Dilation:", _dilation)

    if len(_kernel_size) > 0:
        for dim in range(len(_kernel_size)):
            layer_shape[dim + 1] = (_input_shape[dim + 1] + 2 * _padding[dim] - _dilation[dim] * (layer_shape[dim + 1] - 1) - 1) // _stride[dim] + 1

    print("Output shape:", layer_shape)

def test_set_layer_kernel_size():

    ctx.Net.Init.Layers = [Layer.Def([10, 0, 0])]
    #ctx.Net.Init.Layers = [Layer.Def([10, 3, 0])]
    #ctx.Net.Init.Layers = [Layer.Def([10, 0, 3])]
    #ctx.Net.Init.Layers = [Layer.Def([10, 3, 3])]

    net = ctx.Net()

    net.layers[0].print()

def test_random_kernel_size(_max = 28,
                            _draws = 1000):

    wheel = Rand.RouletteWheel()

    for i in range(1,_max):
       if i % 2 == 1:
            wheel.add(i, math.exp(-i))

    kernels = {}
    for draw in range(_draws):
        k = wheel.spin()

        if k not in kernels.keys():
            kernels[k] = 0
        kernels[k] += 1

    for key in sorted(kernels):
        print(key, ":", kernels[key])

def test_extract_subkernel(_layer_shape = [1,9,9],
                           _input_shape = [1, 28, 28],
                           _node_index = 0,
                           _patch_size = [3, 3]):

    layer = Layer(Layer.Def(_layer_shape), _input_shape)

    layer.print()

    sub = layer.extract_patch(_node_index, _patch_size)

    print(sub)

def test_overlay_kernels():

    layer = Layer(ctx.Net.Init.Layers[0], ctx.Net.Input.Shape)
    print("================[ Initial layer ]================")
    layer.print()
    print("Weight tensor:\n", layer.weight)

    layer.overlay_kernels()
    print("================[ After overlaying kernels ]================")
    layer.print()
    print("Weight tensor:\n", layer.weight)

def test_single_mutation(_mut = 'add_layer'):

    net = ctx.Net()
    net.print('before_mutation.txt', True)
    if _mut == 'add_layer':
        success = net.add_layer()
    elif _mut == 'erase_layer':
        success = net.erase_layer()
    elif _mut == 'add_node':
        success = net.add_nodes()
    elif _mut == 'erase_node':
        success = net.erase_nodes()
    elif _mut == 'grow_kernel':
        success = net.grow_kernel()
    elif _mut == 'shrink_kernel':
        success = net.shrink_kernel()

    else:
        print("Invalid mutation type %r" % _mut)
        return

    assert(pass_fail(success, "Mutating network..."))

    net.print('after_mutation.txt', True)

    model = net.to('cpu')

    tensor = torch.randn(ctx.BatchSize, *ctx.Net.Input.Shape)
    print("Input size:", tensor.size())
    output = model(tensor)
    print(output)

def test_tensor_slices():

    tensor = torch.zeros(5,3,3)
    new_tensor = torch.Tensor()

    print(tensor)
    print(new_tensor)

    slices = [slice(0,2), slice(3,None)]
    for slice_index in range(len(slices)):
        if slice_index == 0:
            new_tensor = tensor[slices[slice_index]]
        else:
            new_tensor = torch.cat((new_tensor, tensor[slices[slice_index]]))

    print(tensor)
    print(new_tensor)

def test_mutations():

    for i in range(100):

        wheel = RouletteWheel()
        wheel.add('add_layer', 1)
        wheel.add('add_node', 1)
        wheel.add('grow_kernel', 1)

        net = ctx.Net()
        original_net = copy.deepcopy(net)
        mutations = []

        for mut in range(100):
            mutation = wheel.spin()
            success = False
            if mutation == 'add_layer':
                success, layer = net.add_layer(_test = True)
                if success:
                    mutations.append((mutation, layer))
            elif mutation == 'add_node':
                success, layer, nodes = net.add_nodes()
                if success:
                    mutations.append((mutation, layer, nodes))
            elif mutation == 'grow_kernel':
                success, layer, kernel, delta = net.grow_kernel()
                if success:
                    mutations.append((mutation, layer, kernel, delta))

            if success:
                print("(", mut + 1, ") Mutation:", *mutations[-1])
            model = net.to('cpu')
            assert(model(torch.randn(ctx.BatchSize, *ctx.Net.Input.Shape)).size())

        print("\n==============[ Reversing mutations ]==============\n")

        for mut in range(len(mutations)):
            mutation = mutations[len(mutations) - mut - 1]
            success = False
            if mutation[0] == 'add_layer':
                success, layer = net.erase_layer(mutation[1])
                #if success:
                    #print("Layer", layer, "erased")
            elif mutation[0] == 'add_node':
                success, layer, node = net.erase_nodes(mutation[1], _node_indices = mutation[2])
                #if success:
                    #print("Node", *nodes, "erased from layer", layer)
            elif mutation[0] == 'grow_kernel':
                success, layer, kernel, delta = net.shrink_kernel(mutation[1], mutation[2], mutation[3])
                #if success:
                    #print("Dimension", *delta.keys(), "of kernel", kernel, "in layer", layer, "decreased by", abs(*delta.values()))

            assert (pass_fail(success, "Reversing mutation", len(mutations) - mut, "(", mutation, ")..."))
            model = net.to('cpu')
            output = model(torch.randn(ctx.BatchSize, *ctx.Net.Input.Shape))

        assert (pass_fail(net.matches(original_net), "Comparing the original network with the one with reversed mutations..."))

        print("======================[ Original network ]======================")
        original_net.print()

        print("======================[ Mutated network ]=======================")
        net.print()

        model1 = net.to('cpu')
        model2 = original_net.to('cpu')

        input1 = torch.randn(ctx.BatchSize, *ctx.Net.Input.Shape)
        input2 = copy.deepcopy(input1)

        print("Input1 size:", input1.size())
        print("Input2 size:", input2.size())

        output1 = model1(input1)
        output2 = model2(input2)

        assert(pass_fail(torch.allclose(output1, output2), "Comparing the two outputs..."))

        print(output1)
        print(output2)

def test_crossover():

    wheel = RouletteWheel()
    wheel.add('add_layer', 1)
    wheel.add('erase_layer', 1)
    wheel.add('add_node', 1)
    wheel.add('erase_node', 1)
    wheel.add('grow_kernel', 1)
    wheel.add('shrink_kernel', 1)

    nets = [ctx.Net() for _ in range(10)]

    for mut in range(10):
        for n in range(len(nets)):
            mutation = wheel.spin()

            net = nets[n]

            if mutation == 'add_layer':
                net.add_layer()

            elif mutation == 'add_node':
                net.add_nodes()

            elif mutation == 'grow_kernel':
                net.grow_kernel()

            if mutation == 'erase_layer':
                net.erase_layer()

            elif mutation == 'erase_node':
                net.erase_nodes()

            elif mutation == 'shrink_kernel':
                net.shrink_kernel()

            model = net.to('cpu')
            match = list(model(torch.randn(ctx.BatchSize, *ctx.Net.Input.Shape)).size()) == [ctx.BatchSize, net.layers[-1].get_output_nodes()]
            if not pass_fail(match, "\tEvaluating the mutated network with random input..."):
                net.print()

    for p1 in range(len(nets)):
        for p2 in range(len(nets)):

            if p1 != p2:

                offspring = ctx.Net(_p1 = nets[p1], _p2 = nets[p2])

                model = offspring.to('cpu')
                match = list(model(torch.randn(ctx.BatchSize, *ctx.Net.Input.Shape)).size()) == [ctx.BatchSize, net.layers[-1].get_output_nodes()]
                if not pass_fail(match, "\tEvaluating the offspring network with random input..."):
                    offspring.print()

def test_init_population():

    ctx.Species.Enabled = False
    ctx.init()

"""
Print a header with some information and run a unit test.
"""

def run(_func,
        *_args,
        **_keywords):

    print("\n==================================[ Unit test ]==================================")
    print("Function:", _func.__name__)
    print("Arguments:")
    for arg in _args:
        print("\t", arg)
    print("Keyword arguments:")
    for key, val in _keywords.items():
        print("\t", key, ":", val)

    if ctx.pause() == 'Y':
        print("Function output:\n\n")
        _func(*_args, **_keywords)
    print("\n===============================[ End of unit test ]==============================")

#run(test_fmap)

#run(test_stat)

#run(test_rand)

#run(test_torch_allclose)

#run(test_print_module_params)

#run(test_init_tensor)

#run(test_output_shape)

#run(test_set_layer_kernel_size)

#run(test_random_kernel_size, 10, 10000)

#run(test_extract_subkernel)

#run(test_overlay_kernels)

#run(test_single_mutation)

#run(test_tensor_slices)

#run(test_mutations)

#run(test_crossover)

run(test_init_population)
