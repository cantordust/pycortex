#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:33:58 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import torch
import torch.nn as tn
import torch.nn.functional as tnf

torch.set_printoptions(precision = 4, threshold = 5000, edgeitems = 5, linewidth = 160)

from kernel import kernel as knl
from kernel.network import Net
from kernel.layer import Layer

import random
import copy

random.seed()

#import knl.rand as Rand
#import knl.statistics as Stat
#import knl.functions as Func

#
#def test_conf():
#    print("\n===[ test_conf() ]===")
#    print(Net)
#    print(Net.Init.Count)
#
#def test_func(_val):
#    print("\n===[ test_func(", _val, ") ]===")
#    for enum, f in Func.fmap.items():
#        print(enum.name, f(_val))
#
#def test_stat(_type = Stat.MAType.Simple):
#    print("\n===[ test_stat(", _type.name, ") ]===")
#    stat = Stat.SMAStat() if _type == Stat.MAType.Simple else Stat.EMAStat()
#    for i in range(1, 100):
#        stat.update(i)
#    print ("mean:", stat.mean, "\nsd:", stat.sd(), "\nalpha:", stat.alpha, "\nvalue:", stat.value)
#
#def test_rand():
#    print("\n===[ test_rand() ]===")
#    print("ND:", Rand.ND())
#    print("negND:", Rand.negND())
#    print("posND:", Rand.posND())
#    print("ureal(-100.0,100.0):", Rand.ureal(-100.0, 100.0))
#    print("uint(-100,100):", Rand.uint(-100, 100))
#    array = [1,2,3,4,5]
#    weights = [5, 3, 100, 67, 22]
#    table = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
#    print("elem:", Rand.elem(array))
#    print("roulette:", Rand.roulette(array, weights))
#    print("key:", Rand.key(table))
#    print("val:", Rand.val(table))


#print("\n======================================\n")

#test_conf()

#print("\n======================================\n")

#test_func(1.0)

#print("\n======================================\n")

#test_stat(knl.MA.Simple)

#print("\n======================================\n")

#test_stat(knl.MA.Exponential)

#print("\n======================================\n")

#test_rand()

#print("\n======================================\n")
#
#print([0, *[i for i in range(1,4)]])
#
#print("\n======================================\n")
#
#l1 = tn.Linear(1,1)
#l2 = tn.Conv1d(1,2,3)
#
#print("type(l1) =", type(l1), ", type(l2) =", type(l2))
#print("type(l1) = type(l2)" if type(l1) == type(l2) else "type(l1) =/= type(l2)")
#
#print("\n======================================\n")
#
#layer = tn.Conv2d(2,3,3)
#
#print(layer)
#
#print("layer parameters:\n", layer)
#for param in layer.parameters():
#    print(param)
#
#layer.add_module("pool", tn.MaxPool2d(2))
#print(layer.pool)
#
#for mod in layer.children():
#    print("\t>>> Child module:", mod)
#
#print("layer parameters:\n", layer)
#for param in layer.parameters():
#    print(param)
#
#layer.add_module("pool", tn.Conv2d(3,2,3))
#
#for mod in layer.children():
#    print("\t>>> Child module:", mod)
#
#print("layer parameters:\n", layer)
#for param in layer.parameters():
#    print(param)
#
#layer.func = tn.Tanh
#
#print("\t>>> Activation:", layer.func)
#
#print("\n======================================\n")
#
#layer = tn.Conv2d(2,3,3, bias = False)
#
#print(layer)
#
#print("layer parameters:\n", layer)
#for param in layer.parameters():
#    print(param)
#
#print("\n======================================\n")
#
#layer = tn.Conv2d(3,3,3)
#init_args = {}
#
##init_type = tn.init.normal_
#
#init_type = tn.init.constant_
#init_args = {'val': 4}
#
#print(init_args)
#args = inspect.signature(init_type)
#missing_args = []
#for param in args.parameters.values():
#    if (param.default is param.empty and
#    param.name != 'tensor' and
#    not param.name in init_args.keys()):
#        missing_args.append(param)
#
#if len(missing_args) == 0:
#    init_type(layer.weight, **init_args)
#else:
#    print(">>> Warning: tensor at layer not initialised due to missing arguments")
#
#print(layer.weight)
#
#print("\n======================================\n")
#
#list1 = [1,2,3,4]
#list2 = [2,2,3,4]
#
#print(list1[1:])
#
#tpl = (list1[1:])
#
#print(tpl)
#
#print("\n======================================\n")
#
#layerLin = tn.Linear(3, 10,)
#layerc1d = tn.Conv1d(3, 10, [5])
#layerc2d = tn.Conv2d(3, 10, [5, 6])
#layerc3d = tn.Conv3d(3, 10, [5, 6, 7])
#
#print(layerLin.weight.size())
#print(layerc1d.weight.size())
#print(layerc2d.weight.size())
#print(layerc3d.weight.size())
#
#print("\n======================================\n")
#
#input_shape = [3]
#
#nodes = 10
#
#if len(input_shape) == 1:
#    layer = tn.Linear(input_shape[0], nodes)
#
#elif len(input_shape) == 2:
#    layer = tn.Conv1d(input_shape[0], 10, (3))
#
#elif len(input_shape) == 3:
#    layer = tn.Conv2d(input_shape[0], 10, (3, 3))
#
#elif len(input_shape) == 4:
#    layer = tn.Conv3d(input_shape[0], 10, (3, 3, 3))
#
#layer_shape = list(layer.weight.size())
#
#del layer_shape[1] # Input channels
#
#print(layer_shape)
#
#if len(layer_shape) > 1:
#    for dim in range(1, len(layer_shape)):
#        layer_shape[dim] = (input_shape[dim] + 2 * layer.padding[dim - 1] - layer.dilation[dim - 1] * (layer_shape[dim] - 1) - 1) // layer.stride[dim - 1] + 1
#
#
#
#print(layer_shape)
#print(len(layer.weight.size()))
#
#print("\n======================================\n")
#wheel = mRand.Roulette()
#
#for i in range(1,21):
##    if i % 2 == 1:
#    wheel.add(i, math.exp(-i))
##    for j in range(1,32):
##        for k in range(1,32):
##            wheel.add(str(i) + ',' + str(j) + ',' + str(k), math.exp(-i) * math.exp(-j) * math.exp(-k))
#
#kernels = {}
#for draw in range(0, 100):
#    k = wheel.spin()
##    k = math.
#    if k not in kernels.keys():
#        kernels[k] = 0
#    kernels[k] += 1
#
#for key in sorted(kernels):
#    print(key, ":", kernels[key])
#
#print("\n======================================\n")
#
#def print_summary(_layer):
#    print(">>> weight:\n", _layer.weight)
#    print(">>> weight size", _layer.weight.size())
#    print(">>> kernel_size", _layer.kernel_size)
#    print(">>> kernels:", _layer.kernels)
#
#layer = tn.Conv3d(2,3,(1,3,5))
#
#layer.kernels = [[1,3,3], [1,1,3], [1,3,1]]
#
#Layer.init(layer)
#
#print_summary(layer)
#
#kernel_idx = 0
#dim_idx = 0
#change = 2
#
#kernel_idx, dim_idx, change = Layer.resize_kernel(layer, kernel_idx, dim_idx, change)
#
#print_summary(layer)
#
## Resize a kernel
#
#kernel_idx = None
#dim_idx = None
#change = None
#
#kernel_idx, dim_idx, change = Layer.resize_kernel(layer, kernel_idx, dim_idx, change)
#print(layer.weight[kernel_idx])

#print("\n======================================\n")
#layer = tn.Conv2d(2,3,(4,5), padding = (1,1))
#
#print(layer)
#
#layer.padding = (2,2)
#
#print(layer)
#
#print("\n======================================\n")
#
## Add an output node
#
#layer = tn.Conv2d(2,3,(3,3))
#print(layer)
#print(layer.weight)
#print(layer.weight.size())
#
#layer.weight = tn.Parameter(torch.cat((layer.weight, torch.zeros(1, *layer.weight[0].size()))))
#layer.out_channels += 1
#print(layer)
#print(layer.weight)
#print(layer.weight.size())

#print("\n======================================\n")
#
## Add an input node
#
#layer = tn.Conv2d(2,3,(3,3))
#print(layer)
#print(layer.weight)
#print(layer.weight.size())
#
#layer.weight = tn.Parameter(torch.cat((layer.weight, torch.zeros(layer.weight.size(0), 1, *layer.weight[0][0].size())), 1))
#layer.in_channels += 1
#print(layer)
#print(layer.weight)
#print(layer.weight.size())randurandu
#
#print("\n======================================\n")
#
#tensor = torch.randn(1,2,3)
#tz = torch.zeros(2,2,3)
#print(tensor)
#tz[0:tensor.size(0)] = tensor
#
#print(tz)
#
#tn.init.normal_(tz[-1])
#print(tz)
#
#print("\n======================================\n")
#
#print(*[0,0])
#
#print([*[0,0]])
#
#print("\n======================================\n")
#
#layer = tn.Conv2d(2,3,(3,3))
#
#layer.kernels = [[3,3], [3,3]]
#
#print(layer.kernels)
#
#kernel = layer.kernels[0]
#
#kernel[0] -= 2
#
#print(layer.kernels)
#print("\n======================================\n")

#tensor = torch.randn(2,3,4)
#
#print(tensor)
#
#tensor = torch.cat((tensor, torch.zeros(tensor.size(0), *tensor[0].size())), dim = 1)
#
#print(tensor)
#
#layer = tn.Conv2d(2,3,(5,5))
#
#print(layer.weight)
#print(torch.zeros(layer.weight[0].size()))
#
#layer.weight = tn.Parameter(torch.cat((layer.weight, torch.zeros(layer.weight.size(0), 1, *layer.weight.size()[2:])), dim = 1))
#print(layer.weight)
#
#print(list(layer.weight[-1][0].size()))
#
#for kernel in layer.weight:
#    Func.init_tensor(kernel[-1], tn.init.uniform_, {'a': -1, 'b': 1})
#
#print(layer.weight)

#print("\n======================================\n")
#max_kernel_size = [15,15,15]
#
#kernels = Layer.get_random_kernel_size(max_kernel_size, 10)
#print(kernels)
#
#print("\n======================================\n")
#
#src = tn.Parameter(torch.ones(1,3,7))
#tgt = tn.Parameter(torch.zeros(3,3,5))
#
#tgt.data.fill_(2.0)
#
#Layer.overlay_kernels(src, tgt)
#
#print(src)
#print(tgt)
#
#sub = Layer.extract_subkernel(tgt, [3,3,3])
#
#print(sub)

#
#print("\n======================================\n")
#
#src = torch.randn(1,3,5)
#
#tgt = Layer.extract_subkernel(src, [1,1,3])
#
#print(src)
#print(tgt)

#print("\n======================================\n")
#
#src = torch.randn(1,3,5)
#
#centre = Layer.get_centre(src.size())
#
#print(centre)
#print("\n======================================\n")
#
#lst = [1,2,3]
#
#tpl = tuple([x // 2 for x in lst])
#
#print(tpl)
#print("\n======================================\n")
#tensor = torch.zeros(1,2,3)
#
#print(tensor)
#
#Func.init_tensor(tensor, tn.init.uniform_, {'a': -1, 'b': 1})
#
#print(tensor)
#
#print("\n======================================\n")
#
#layer = tn.Conv2d(2,3,(3,3), bias = False)
#
#print(layer.bias)
#
#for param in layer.parameters():
#    print(param)
#
#Func.init_tensor(layer.weight, tn.init.zeros_)
#
#for param in layer.parameters():
#    print(param)
#
#print("\n======================================\n")

#net = knl.mNet.Net()
#
#knl.mNet.print_net(net)
#
#print(net.layers[0].weight)
#print(net.layers[0].mask)
#
#for n in range(100):
#    net.resize_kernel(0)
#
#print(net.layers[0].weight)
#print(net.layers[0].mask)
#
#Layer.apply_mask(net.layers[0])
#
#print(net.layers[0].weight)
#print(net.layers[0].kernels)
#
#net.add_nodes(0)
#Layer.apply_mask(net.layers[0])
#
#knl.mNet.print_net(net)

#net.add_nodes(0, 2)
#print("\n=================================\n")
#print("After adding nodes")
#print("\n=================================\n")
#knl.mNet.print_net(net)
#
#net.add_nodes(1, 2)
#print("\n=================================\n")
#print("After adding nodes")
#print("\n=================================\n")
#knl.mNet.print_net(net)
#
#net.add_nodes(1, 2)
#print("\n=================================\n")
#print("After adding nodes")
#print("\n=================================\n")
#knl.mNet.print_net(net)

#for param in net.parameters():
#    print(">>> param:\n", param)
#
#print("\n======================================\n")
#
#net = knl.mNet.Net()
#
#knl.mNet.print_net(net)
#
#net.resize_kernel(0)
#
#knl.mNet.print_net(net)

#print("\n======================================\n")
#
#tensor1 = torch.randn(1,3,3)
#
#tensor2 = torch.randn(1,3,3)
#
#print(tensor1)
#print(tensor2)
#
#layer = tn.Conv3d(1,2,(1,3,5))
#
#layer.weight.data.fill_(0.0)
#
#
#print(layer.weight)
#
#Layer.overlay_kernels(tensor1, layer.weight.data[0][0])
#Layer.overlay_kernels(tensor2, layer.weight.data[1][0])
#
#print(layer.weight)
#
#layer.weight.data.fill_(0.0)
#
#print(tensor1)
#print(tensor2)
#print(layer.weight)
#
#print("\n======================================\n")
#
#layer = tn.Linear(5,1, bias = True)
#for param in layer.parameters():
#    param.requires_grad = False
#layer.bias.requires_grad = True
#
#layer.kernels = tn.ParameterList()
#layer.kernels.append(tn.Parameter(torch.randn(3)))
#
#for n in range(layer.weight.size(0)):
#    layer.weight[n].data[0].fill_(0.0)
#    layer.weight[n].data[-1].fill_(0.0)
#
#print(layer.weight)
#
#optimizer = torch.optim.Adadelta(layer.parameters())
#
#for param in layer.parameters():
#    print(param)
#
#target = torch.randn(1, 1)
#
#for n in range(10):
#    optimizer.zero_grad()
#    layer.weight[0][1:4] = layer.kernels[0]
#    output = layer.forward(torch.randn(1,5))
#    loss = tnf.mse_loss(output, target)
#    loss.backward(retain_graph = True)
#    optimizer.step()
#    print(">>> Loss:", loss.item())
#    print(">>> Gradient tensor:\n", layer.kernels[0].grad)
#    print(">>> Weights:\n", layer.weight)
#    print(">>> Updated weights:\n", layer.kernels[0])

#print("\n======================================\n")
#
#net = knl.mNet.Net()
#knl.mNet.print_net(net)
#
#net.erase_layer(1)
#knl.mNet.print_net(net)
#
#net.add_layer(_shape = [10], _layer_idx = 1)
#knl.mNet.print_net(net)
#
#print("\n======================================\n")
#
#net = knl.mNet.Net()
#knl.mNet.print_net(net)
#
#print(">>> Layers:", len(net.layers))
#
#for n in range(100):
#    layer_idx = knl.mRand.uint(0,len(net.layers) - 1)
#    shape = net.get_output_shape(layer_idx)
#    shape[0] = knl.mRand.uint(1,10)
#    if net.erase_layer(layer_idx):
#        net.add_layer(_layer_idx = layer_idx, _shape = shape)
#
#knl.mNet.print_net(net)
#
#print("\n======================================\n")
#
#layer = tn.Conv2d(2,3,(3,3))
#layer.type = layer.__class__.__name__
#
#print(">>> layer.__class__.__name__:", layer.__class__.__name__)
#
#print("layer.type == tn.Conv2d:", layer.type == tn.Conv2d)
#print("layer.type == \"Conv2d\":", layer.type == "Conv2d")
#
#print("\n======================================\n")
#
#p1 = knl.Net()
#p1.add_layer(_shape = [5, 0, 0], _layer_index = 1)
#p2 = knl.Net()
#p2.add_nodes(2, 2)
#
#index = 2
#
#p1.add_layer(_shape = [10], _layer_index = 3)
#
#offspring = knl.Net(_p1 = p1, _p2 = p2)
#offspring.print()
#
#f_p1 = None
#f_p2 = None
#f_os = None

#f_p1 = open("p1.txt", 'w')
#f_p2 = open("p2.txt", 'w')
#f_os = open("offspring.txt", 'w')
#
#f_p1 = open("p1.txt", 'w')
#f_p2 = open("p2.txt", 'w')
#f_os = open("offspring.txt", 'w')
#
#f_p1.truncate()
#f_p2.truncate()
#f_os.truncate()

#p1.print(_file = f_p1)
#p2.print(_file = f_p2)
#offspring.print(_file = f_os)

#f_p1.close()
#f_p2.close()
#f_os .close()

#print("\n======================================\n")
#
#knl.init()
#
#print("\n======================================\n")
#
#net = knl.mNet.Net()
#knl.mNet.Net.population[net.ID] = net
#
#net.mutate()
#print("\n======================================\n")
#layer = tn.Conv2d(1,5,(3,3))
#print(layer.weight.size())
#new_weight = torch.squeeze(layer.weight)
#print(new_weight.size())
#print("\n======================================\n")
#
#tensor = torch.ones(1,3,3)
##padding = [1,1,0,0,0,0]
#padding  = [0,0,1,1,0,0]
##padding = [0,0,0,0,1,1]
#print(tensor)
#tensor = tnf.pad(tensor, padding)
#print(tensor)
#
#print("\n======================================\n")
#
#layer = Layer(knl.Net.Init.Layers[0], knl.Net.Input.Shape)
#print("================[ Initial layer ]================")
#layer.print()
#
#layer.overlay_kernels()
#print("================[ After overlaying kernels ]================")
#layer.print()
#
#print("================[ Before resizing kernel ]================")
#layer.print()
#layer.resize_kernel(0)
##layer.overlay_kernels()
#print("================[ After resizing kernel ]================")
#layer.print()
#
#print("================[ Before resizing layer ]================")
#layer.print()
#layer.resize(-3)
#print("================[ After resizing layer ]================")
#layer.print()

#print("\n======================================\n")

#for i in range(100):
    #net = knl.Net()
    ##net.print('before_mutation.txt', True)
    ##success = net.add_layer()
    ##success = net.erase_layer()
    ##success = net.add_nodes()
    ##success = net.erase_nodes()
    #success = net.grow_kernel()
    ##success = net.shrink_kernel()
    ##net.print('after_mutation.txt', True)
    #model = net.to('cpu')

    #tensor = torch.randn(knl.BatchSize, *knl.Net.Input.Shape)
    #print("Input size:", tensor.size())
    #output = model(tensor)
    #print(output)

#for i in range(100):
    #print("Mutation", i)
    #net.print('before_mutation.txt', True)
    #success = net.mutate()
    #net.print('after_mutation.txt', True)
    #if not success:
        #print("!!! Failed !!!")
        #break
    #model = net.to('cpu')
    #model(torch.randn(knl.BatchSize, *knl.Net.Input.Shape))
    #knl.pause()

#knl.init()

#print("\n======================================\n")

#tensor = torch.zeros(5,3,3)
#new_tensor = torch.Tensor()

#print(tensor)
#print(new_tensor)

#slices = [slice(0,2), slice(3,None)]
#for slice_index in range(len(slices)):
    #if slice_index == 0:
        #new_tensor = tensor[slices[slice_index]]
    #else:
        #new_tensor = torch.cat((new_tensor, tensor[slices[slice_index]]))

#print(tensor)
#print(new_tensor)

#print("\n======================================\n")

#tensor1 = torch.ones(3,3)
#tensor2 = torch.ones(3,3)
#tensor2[0][0] += 1e-8

#print("tensor1:\n", tensor1)
#print("tensor2:\n", tensor2)

#print("tensor1 == tensor2", torch.allclose(tensor1, tensor2, 1e-8, 1e-8))

print("\n======================================\n")

from colorama import Fore, Style
from kernel.rand import RouletteWheel

def pass_fail(cond, *args):
    print(f'[ {Fore.GREEN}Passed{Style.RESET_ALL} ]' if cond else f'[ {Fore.RED}Failed{Style.RESET_ALL} ]', *args)
    return cond

for i in range(100):

    net = knl.Net()
    net_clone = copy.deepcopy(net)
    mutations = []

    wheel = RouletteWheel()
    wheel.add('add_layer', 1)
    #wheel.add('erase_layer', 1)
    wheel.add('add_node', 1)
    #wheel.add('erase_node', 1)
    wheel.add('grow_kernel', 1)
    #wheel.add('shrink_kernel', 1)

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
        assert(model(torch.randn(knl.BatchSize, *knl.Net.Input.Shape)).size())

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
        output = model(torch.randn(knl.BatchSize, *knl.Net.Input.Shape))

    assert (pass_fail(net.matches(net_clone), "Comparing the original network with the one with reversed mutations..."))

    #print("======================[ Original network ]======================")
    #net_clone.print()

    #print("======================[ Mutated network ]======================")
    #net.print()

    model1 = net.to('cpu')
    model2 = net_clone.to('cpu')

    input1 = torch.randn(knl.BatchSize, *knl.Net.Input.Shape)
    input2 = copy.deepcopy(input1)

    print("Input1 size:", input1.size())
    print("Input2 size:", input2.size())

    output1 = model1(input1)
    output2 = model2(input2)

    assert(pass_fail(torch.allclose(output1, output2), "Comparing the two outputs..."))

    print(output1)
    print(output2)

