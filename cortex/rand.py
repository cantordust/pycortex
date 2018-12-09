#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:02:21 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import random
from . import statistics as mStat
from . import functions as mFunc

from enum import Enum

# Returns a random number drawn from a normal distribution.
def ND(_mean = 0.0, _sd = 1.0):
        return random.gauss(_mean, _sd)

# Returns a random number drawn from the positve half of a normal distribution.
def posND(_mean = 0.0, _sd = 1.0):
    return abs(ND(_mean, _sd))

# Returns a random number drawn from the negative half of a normal distribution.
def negND(_mean = 0.0, _sd = 1.0):
    return -abs(ND(_mean, _sd))

# Returns a random floating-point value between _min and _max.
def ureal(_min = 0.0, _max = 1.0):
    return random.uniform(_min, _max)

# Returns the outcome of a condition check with probability @p _prob.
def chance(_prob):
    return ureal(0.0, 1.0) <= _prob

# Returns the outcome of a condition check with probability @p _prob.
def inverse_chance(_prob):
    return chance(1.0 - _prob)

# Roulette wheel selection.
def roulette(_collection, _weights):
    if len(_collection) == 0:
        return None
    return random.choices(_collection, _weights)[0]

# Returns a random integer value in the range [_min, _max).
def uint(_min, _max):
    return random.randint(_min, _max - 1)

# Returns a random element from a container.
def elem(_container):
    if len(_container) == 0:
        return None
    return random.choice([elem for elem in _container])

# Returns a random key from an associative container.
def key(_container):
    if not isinstance(_container, dict) or len(_container) == 0:
        return None
    return random.choice(list(_container.keys()))

# Returns a random value from an associative container.
def val(_container):
    if not isinstance(_container, dict) or len(_container) == 0:
        return None
    return random.choice(list(_container.values()))

class WeightType(Enum):
    Raw = 'Raw'
    Normal = 'Normal'
    Inverse = 'Inverse'

class RouletteWheel:

    def __init__(self,
                 _weight_type = WeightType.Raw,
                 _norm_func = mFunc.Type.Logistic):
        self.elements = []
        self.weight_type = _weight_type
        self.norm_func = _norm_func
        self.is_normalised = False
        self.frozen_elem = None
        self.weights = {
                    WeightType.Raw: [],
                    WeightType.Normal: [],
                    WeightType.Inverse: []
                    }

    def add(self,
            _elem,
            _weight):
        self.elements.append(_elem)
        self.weights[WeightType.Raw].append(_weight)

    def spin(self,
             _weight_type = None):

        if self.is_empty():
            return None

        weight_type = self.weight_type if _weight_type is None else _weight_type

        if self.frozen_elem is not None:
            return self.frozen_elem

        else:
            # Normalise if necessary
            if (weight_type != WeightType.Raw and
                not self.is_normalised):
                self.normalise()

            return roulette(self.elements, self.weights[weight_type])

    def pop(self,
            _weight_type = None):

        if self.is_empty():
            return None

        weight_type = self.weight_type if _weight_type is None else _weight_type

        # Unfreeze the wheel if we are going to modify it
        self.unfreeze()

        # Normalise if necessary
        if (weight_type != WeightType.Raw and
            not self.is_normalised):
                self.normalise()

        # Choose an element index
        indices = [idx for idx in range(len(self.elements))]
        index = roulette(indices, self.weights[weight_type])

        # Store the chosen element
        element = self.elements[index]

        # Delete the element and the corresponding weight
        del self.elements[index]
        del self.weights[WeightType.Raw][index]

        # Indicate that the weights are not normalised
        self.is_normalised = False

        return element

    def replace(self,
                _new_elements = []):
        assert isinstance(_new_elements, list) and len(self.elements) == len(_new_elements), "Invalid element list %r" % _new_elements
        self.elements = list(_new_elements)

    def normalise(self):

        if self.is_empty():
            return

        for weight_type, lst in self.weights.items():
            if weight_type != WeightType.Raw:
                lst.clear()

#        self.weights[WeightType.Normal] = mFunc.scale(self.weights[WeightType.Raw])
        self.weights[WeightType.Normal] = list(self.weights[WeightType.Raw])

        for w in self.weights[WeightType.Normal]:
            self.weights[WeightType.Inverse].append(1 / w if w != 0.0 else 0.0)

        self.is_normalised = True

    def freeze(self):
        if self.frozen_elem is None:
            self.frozen_elem = self.spin()

    def unfreeze(self):
        self.frozen_elem = None

    def is_empty(self):
        return len(self.elements) == 0
