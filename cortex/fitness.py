#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:59:17 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

class Fitness:

    Target = None

    def __init__(self):

        from . import statistics as Stat

        self.absolute = Stat.EMAStat()
        self.relative = Stat.EMAStat()

    def calibrate(self, 
                  _stat):
        
        from . import statistics as Stat
        
        self.relative.update(_stat.get_offset(self.absolute.value))
