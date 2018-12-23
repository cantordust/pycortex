#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:23:52 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import torch
from torchvision import datasets, transforms

from cortex import cortex as ctx



ctx.Net.Input.Shape = [1, 32, 32]
ctx.Net.Output.Shape = [10]
ctx.TrainFunction = train