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

        self.abs = 0.0
        self.rel = 0.0

    def calibrate(self,
                  _stat):

        self.rel = (_stat.get_offset(self.abs))
