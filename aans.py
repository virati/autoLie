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
f = f_main
h = f1

corr_matrix = jcb(f)

readout = L_d(f,h)

x = np.array([1.,1.,1.])

print('Main X')
print(f_main(x))
print('Corr Matrix')
print(corr_matrix(x))



