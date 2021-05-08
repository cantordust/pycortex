import sys
import math
from enum import Enum

import cortex.functions as Func

class MAType(Enum):
     Exponential = 'Exponential'
     Simple = 'Simple'

class Stat:

    def __init__(self,
                 _title = None,
                 _func = Func.Type.Logistic):
        self.title = "Statistics" if _title is None else _title
        self.reset()
        self.func = _func

    def get_sd(self):
        return math.sqrt(self.var)

    def reset(self):
        self.current_value = 0.0
        self.min = math.inf
        self.abs_min = math.inf
        self.max = -math.inf
        self.abs_max = 0.0
        self.mean = 0.0
        self.var = 0.0

    def update(self,
               _new_val):

        self.current_value = _new_val
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
                   _func = None):

        if _val is None:
            _val = self.current_value

        denom = 1.0
        if self.var > 0.0:
            denom = self.get_sd()
        elif _val != 0.0:
            denom = abs(_val)
        elif self.mean != 0.0:
            denom = abs(self.mean)

        return Func.fmap[_func if _func is not None else self.func]((_val - self.mean) / denom)

    def get_inv_offset(self,
                       _val = None):
        return self.get_offset(_val, Func.Type.InvLogistic)

    def restore(self,
                _val = None,
                _func = None):

        if _val is None:
            return None

        func = self.func if _func is None else _func

        if func == Func.Type.Tanh:
            inv_func = 0.5 * math.log( (1.0 + _val) / (1.0 - _val) )
        else:
            # Logistic
            inv_func = math.log(_val / (1.0 - _val))

        return self.get_sd() * inv_func + self.mean

    def as_str(self):

        str = f'\n======[ {self.title} ]======' +\
              f'\nCurrent value: {self.current_value}' +\
              f'\nMean: {self.mean}' +\
              f'\nVariance: {self.var}' +\
              f'\nSD: {self.get_sd()}' +\
              f'\nMinimum: {self.min}' +\
              f'\nMaximum: {self.max}'

        return str

class SMAStat(Stat):

    def __init__(self,
                 _title = None,
                 _func = Func.Type.Logistic):
        super(SMAStat, self).__init__(_title = _title, _func = _func)
        self.count = 0

    def update(self,
               _new_val):
        super(SMAStat, self).update(_new_val)

        self.count += 1
        old_mean = self.mean
        self.mean += (self.current_value - old_mean) / self.count
        if (self.count > 1):
            self.var += (self.current_value - old_mean) * (self.current_value - self.mean) / self.count

    def as_str(self):

        str = super(SMAStat, self).as_str() +\
              f'\nCount: {self.count}'

        return str

class EMAStat(Stat):

    def __init__(self,
                 _alpha = 0.5,
                 _title = None,
                 _func = Func.Type.Logistic):
        super(EMAStat, self).__init__(_title = _title, _func = _func)
        self.alpha = _alpha

    def update(self, _new_val):
        super(EMAStat, self).update(_new_val)

        diff = self.current_value - self.mean
        inc = self.alpha * diff
        self.mean += inc
        self.var = (1.0 - self.alpha) * (self.var + diff * inc)

    def as_str(self):

        str = super(EMAStat, self).as_str() +\
              f'\nAlpha: {self.alpha}'

        return str
