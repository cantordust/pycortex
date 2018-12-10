#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:10:32 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

from enum import Enum
import math

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

def linscale(_list):

    if len(_list) == 0:
        return _list

    from .statistics import SMAStat

    stat = SMAStat()
    
    for elem in _list:
        stat.update(elem)

    scaled = [stat.abs_min + (stat.abs_max - stat.abs_min) * (x - stat.min) / (stat.max - stat.min)]

    #print(scaled)

    return scaled

def softmax(_list):

    if len(_list) == 0:
        return _list

    #_list = linscale(_list)

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

    product = 1
    for elem in _list:
        product *= elem

    return product

def exp_prod(_list):

    product = 1
    for elem in _list:
        product *= math.exp(-elem)

    return product

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
