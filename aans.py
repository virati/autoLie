#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:25:54 2019

@author: virati
AANS Submission
"""

from lie_lib import *
from dyn_lib import *

# First, we set up our dynamics matrix
f = f1
h = h1

corr_matrix = jcb(f)

readout = L_d(f,h)

x = np.array([1.,2.,3.])

print('f')

for ii in np.linspace(-1,1,100):
    for jj in np.linspace(-1,1,100):
        for kk in np.linspace(-1,1,100):
            readout = L_d(f,h)

print('f at x')
print((f(x)))
print('Jacobian of f')
print(corr_matrix(x))
print('h at x')
print(h(x))
print('h along f')
print(readout(x))



