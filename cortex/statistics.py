import sys
import math
from enum import Enum

import cortex.functions as Func

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
                   _func = Func.Type.Logistic):

        if _val is None:
            _val = self.current_value

        denom = 1.0
        if self.var > 0.0:
            denom = self.get_sd()
        elif _val != 0.0:
            denom = abs(_val)
        elif self.mean != 0.0:
            denom = abs(self.mean)

        return Func.fmap[_func]((_val - self.mean) / denom)

    def get_inv_offset(self,
                       _val = None):
        return self.get_offset(_val, Func.Type.InvLogistic)

    def as_str(self):

        str = f'''
======[ {self.title} ]======
Current value: {self.current_value}
Mean: {self.mean}
Variance: {self.var}
SD: {self.get_sd()}
Minimum: {self.min}
Maximum: {self.max}'''

        return str

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
              _file = sys.stdout):

        super(SMAStat, self).print(_file = _file)
        print("Count: %r" % self.count, file = _file)

class EMAStat(Stat):

    def __init__(self,
                 _alpha = 0.25,
                 _title = None):
        super(EMAStat, self).__init__(_title = _title)
        self.alpha = _alpha

    def update(self, _new_val):
        super(EMAStat, self).update(_new_val)

        diff = self.current_value - self.mean
        inc = self.alpha * diff
        self.mean += inc
        self.var = (1.0 - self.alpha) * (self.var + diff * inc)

    def print(self,
              _file = sys.stdout):

        super(EMAStat, self).print(_file = _file)
        print("Alpha: %r" % self.alpha, file = _file)
