#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:00:17 2019

@author: virati
"""

import numpy as np
from lie_lib import *

@operable
def h1(x):
    return np.array([0.0,0.0,1.0]).T * x

#@operable
def f1(x,bifur=0):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])

@operable
def f2(x,bifur=0):
    return np.array([-np.sin(x[1]), -5*x[0]**2, -np.sin(x[2] - x[1])])

#@operable
def f3(x,bifur=0):
    return -np.array([(x[0])*(x[0]-2)*(x[0]+2),x[1]**2,x[2]**2])

#@operable
def f4(x,bifur=0):
    return -np.array([x[0],x[1],x[2]])
#@operable
def f5(x,bifur=0):
    return -np.array([np.sin(x[2]),np.sin(x[0]),np.sin(x[1])])
    #return -np.array([x[0] + x[1], x[1]*x[2],x[2]**2])

#@operable
def f6(x,bifur=[0,1]):
    return -np.array([bifur[0]*x[0],bifur[1]*x[1],x[2]])

#@operable
def f7(x,bifur=[0,1]):
    return -np.array([x[0]**3 - bifur[0]*x[0]**2,x[2]**3 - 2*x[1]**2,x[1]**3 - bifur[1]*x[2]])

#@operable
def f8(x,bifur=[0,0]):
    return -np.array([x[1] * x[2],x[2] * (bifur[0] + x[1] - x[2]),x[1] * (-bifur[1] + x[2] + x[1])])

def f9(x):
    return -np.array([x[1] * x[2],x[2] * (x[1] - x[2]),x[1] * (x[2] + x[1])])


@operable
def f_main(x):
    return np.array([[-x[0],-x[1]*x[0],0],[-x[1]*x[2],-x[1],0],[0,-x[0]*x[1],x[2]]])
