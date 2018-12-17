#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:36:09 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import sys
import math
from enum import Enum
from cortex import functions as Func

class MAType(Enum):
     Exponential = 'Exponential'
     Simple = 'Simple'

class Stat:

    def __init__(self,
                 _title = None):
        self.title = "Statistics" if _title is None else _title
        self.reset()

    def get_sd(self):
        return math.sqrt(self.var)

    def reset(self):
        self.current_value = 0.0
        self.sum = 0.0
        self.min = math.inf
        self.abs_min = math.inf
        self.max = -math.inf
        self.abs_max = 0.0
        self.mean = 0.0
        self.var = 0.0

    def update(self,
               _new_val):

        self.current_value = _new_val
        self.sum += _new_val
        if _new_val < self.min:
            self.min = _new_val
        if abs(_new_val) < self.abs_min:
            self.abs_min = abs(_new_val)
        if _new_val > self.max:
            self.max = _new_val
        if abs(_new_val) > self.abs_max:
            self.abs_max = abs(_new_val)

    def get_offset(self,
                   _val = None,
                   _func = Func.Type.Logistic):

        val = self.current_value if _val is None else _val
        denom = 1.0
        if self.var > 0.0:
            denom = self.get_sd()
        elif val != 0.0:
            denom = abs(val)
        elif self.mean != 0.0:
            denom = abs(self.mean)

        return Func.fmap[_func]((val - self.mean) / denom)

    def get_inv_offset(self,
                       _val = None):
        return self.get_offset(_val, Func.Type.InvLogistic)

    def print(self,
              _file = sys.stdout,
              _truncate = True):

        if isinstance(_file, str):
            _file = open(_file, 'w')
            if _truncate:
                _file.truncate()

        print("\n\n======[", self.title, "]======",
              "\nCurrent value: %r" % self.current_value,
              "\nSum: %r" % self.sum,
              "\nMean: %r" % self.mean,
              "\nVariance: %r (SD: %r)" % (self.var, self.get_sd()),
              "\nMinimum: %r" % self.min,
              "\nMaximum: %r" % self.max,
              file = _file)

class SMAStat(Stat):

    def __init__(self,
                 _title = None):
        super(SMAStat, self).__init__(_title = _title)
        self.count = 0

    def update(self,
               _new_val):
        super(SMAStat, self).update(_new_val)

        self.count += 1
        old_mean = self.mean
        self.mean += (self.current_value - old_mean) / self.count
        if (self.count > 1):
            self.var += (self.current_value - old_mean) * (self.current_value - self.mean) / self.count

    def print(self,
              _file = sys.stdout,
              _truncate = True):

        if isinstance(_file, str):
            _file = open(_file, 'w')
            if _truncate:
                _file.truncate()

        super(SMAStat, self).print(_file = _file)
        print("Count: %r" % self.count, file = _file)

class EMAStat(Stat):

    def __init__(self,
                 _alpha = 0.25,
                 _title = None):
        super(EMAStat, self).__init__(_title = _title)
        self.alpha = _alpha

    def update(self, _new_val):
        super(SMAStat, self).update(_new_val)

        diff = self.current_value - self.mean
        inc = self.alpha * diff
        self.mean += inc
        self.var = (1.0 - self.alpha) * (self.var + diff * inc)

    def print(self,
              _file = sys.stdout,
              _truncate = True):

        if isinstance(_file, str):
            _file = open(_file, 'w')
            if _truncate:
                _file.truncate()

        super(SMAStat, self).print(_file = _file)
        print("Alpha: %r" % self.alpha, file = _file)
