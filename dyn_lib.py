#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:00:17 2019

@author: virati
"""

import numpy as np
from lie_lib import *

@operable
def f1(x):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])

@operable
def f2(x):
    return np.array([-np.sin(x[1]), -5*x[0]**2, -np.sin(x[2] - x[1])])

@operable
def f3(x):
    return -np.array([(x[0])*(x[0]-2)*(x[0]+2),x[1]**2,x[2]**2])

@operable
def f4(x):
    return -np.array([x[0],x[1],x[2]])
@operable
def f5(x):
    return -np.array([np.sin(x[2]),np.sin(x[0]),np.sin(x[1])])
    #return -np.array([x[0] + x[1], x[1]*x[2],x[2]**2])
@operable
def f6(x):
    return -np.array([x[0],x[1],x[2]])

@operable
def f7(x,c=4):
    return -np.array([x[0]**3 - c*x[0]**2,x[2]**3 - 2*x[1]**2,x[1]**3 - 3*x[2]])

#@operable
def f8(x,bifur):
    return -np.array([x[1] * x[2],x[2] * (bifur + x[1] - x[2]),x[1] * (-1.0 + x[2] + x[1])])
