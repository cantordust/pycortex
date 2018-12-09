#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:10:32 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

from enum import Enum
import math

#import inspect
#import torch.nn as tn

class Type(Enum):
     Abs = 'Abs'
     Sin = 'Sin'
     Cos = 'Cos'
     Gaussian = 'Gaussian'
     Logistic = 'Logistic'
     InvLogistic = 'Inverse logistic'
     SQRL = 'SQRL'
     Tanh = 'Tanh'

def logistic(_val):
    return 0.5 * (math.tanh( 0.5 * _val ) + 1.0)

def inv_logistic(_val):
    return (1.0 - logistic(_val))

def gaussian(_val):
    return math.exp(-0.5 * math.pow(_val, 2))
#
#def scale(_list):
#
#    if len(_list) < 2:
#        return _list
#
#    max_elem = -math.inf
#    min_elem = math.inf
#
#    for elem in _list:
#        if elem > max_elem:
#            max_elem = elem
#        if elem < min_elem:
#            min_elem = elem
#
#    diffs = [_list[idx] - _list[idx - 1] for idx in range(1,len(_list))]
#
#    max_diff = -math.inf
#    min_diff = math.inf
#
#    for diff in diffs:
#        if diff > max_diff:
#            max_diff = diff
#        if diff < min_diff:
#            min_diff = diff
#
#    if (max_diff == 0.0 or
#        min_diff == 0.0):
#        return _list
#
#    scaled = []
#
#    # ???
#
#    print(scaled)
#
#    return scaled

#def scale(_list):
#
#    if len(_list) == 0:
#        return _list
#
#    from .statistics import SMAStat
#
#    elem_stat = SMAStat(mFunc.Type.Tanh)
#
#    for elem in _list:
#        elem_stat.update(elem)
#
#    # If all the elements are the same,
#    # they should have equal weights.
#    if elem_stat.min == elem_stat.max:
#        return [1 / len(_list)] * len(_list)
#
#    order_diff = math.log(stat.max if stat.max != 0.0 else 1.0) - math.log(stat.min if stat.min != 0.0 else 1e-10)
#
#    # Scaled elements
#    scaled = []
#
#    for elem in _list:
#        if stat.get_offset(elem) > 0.5:
#            scaled.append(math.pow(10, stat.get_offset(elem, ) * order_diff))
#
#        else:
#
#
#    print(scaled)
#
#    return scaled

def softmax(_list):

    if len(_list) == 0:
        return _list

    _list = scale(_list)

    normalised = []

    max_elem = -math.inf

    for elem in _list:
        if elem > max_elem:
            max_elem = elem

    total = 0.0
    for elem in _list:
        normalised.append(math.exp(elem - max_elem))
        total += normalised[-1]

    for idx in range(len(normalised)):
        normalised[idx] /= total

    return normalised

# Differentiable ReLU passing through the origin.
def sqrl(_val):
    return (0.5 * (math.sqrt(math.pow(_val, 2) + 4) + _val) - 1.0)

def prod(_list):

    #if len(_list) == 0:
        #return 0

    product = 1
    for elem in _list:
        product *= elem

    return product

def exp_prod(_list):

    product = 1
    for elem in _list:
        product *= math.exp(-elem)

    return product

def is_almost_equal(_tensor1,
                    _tensor2,
                    _eps = 1.0e-8):

    if _tensor1.size() != _tensor2.size():
        return False

    for elem in (_tensor1 - _tensor2).view(-1):
        if abs(elem) > _eps:
            return False

    return True

def init_tensor(_tensor):

    from .network import Net

    if (_tensor is None or
        not callable(Net.Init.Func)):
        return False

#    expected_args = inspect.signature(mConf.Net.Init.Func)
#    missing_args = []
#    for param in expected_args.parameters.values():
#        if (param.default is param.empty and
#            param.name != 'tensor' and
#            not param.name in mConf.Net.Init.Args.keys()):
#            missing_args.append(param.name)
#
#    if len(missing_args) > 0:
#        print(missing_args)
#        return False
#
#    if (mConf.Net.Init.Func.__class__ == tn.init.xavier_normal_ or
#        mConf.Net.Init.Func.__class__ == tn.init.xavier_uniform):
    Net.Init.Func(_tensor, **Net.Init.Args)

#    print(">>> Tensor initialised with", mConf.Net.Init.Func.__name__)

    return True

fmap = {
    Type.Abs: abs,
    Type.Cos: math.cos,
    Type.Gaussian: gaussian,
    Type.Logistic: logistic,
    Type.InvLogistic: inv_logistic,
    Type.SQRL: sqrl,
    Type.Sin: math.sin,
    Type.Tanh: math.tanh
}
