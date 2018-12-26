from enum import Enum
import math

import torch
import torch.nn as tn

class Type(Enum):
     Abs = 'Abs'
     Sin = 'Sin'
     Cos = 'Cos'
     Gaussian = 'Gaussian'
     Logistic = 'Logistic'
     InvLogistic = 'Inverse logistic'
     SQRL = 'SQRL'
     Tanh = 'Tanh'
     Softmax = 'Softmax'

def logistic(_val):
    return 0.5 * (math.tanh( 0.5 * _val ) + 1.0)

def inv_logistic(_val):
    return (1.0 - logistic(_val))

def gaussian(_val):
    return math.exp(-0.5 * math.pow(_val, 2))

def linscale(_list):

    if len(_list) == 0:
        return _list

    import cortex.statistics as Stat

    stat = Stat.SMAStat()

    for elem in _list:
        stat.update(elem)

    # Scale between 0 and the absolute maximum
    scaled = [stat.abs_max * (x - stat.min) / (stat.max - stat.min) for x in _list]

    #print(scaled)

    return scaled

def softmax(_list):

    if len(_list) == 0:
        return _list

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
    return (0.5 * (math.sqrt(math.pow(_val, 2.0) + 4.0) + _val) - 1.0)

class SQRL(tn.Module):

    def __init__(self):
        super(SQRL, self).__init__()

    def forward(self, x):
        return (0.5 * (torch.sqrt(torch.pow(x, 2.0) + 4.0) + x) - 1.0)

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
    Type.Tanh: math.tanh,
    Type.Softmax: softmax
}
