#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:47:58 2018

@author: virati
Lie Derivatives on predefined operators
"""

from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import grad
from autograd import elementwise_grad as egrad
from autograd import jacobian as jcb
from functools import reduce


##
import operator

npr.seed(1)

# Generate a class that wraps functions
class operable:
    def __init__(self, f):
        self.f = f
    def __call__(self, x):
        return self.f(x)
 
# Convert oeprators to the corresponding function
def op_to_function_op(op):
    def function_op(self, operand):
        def f(x):
            return op(self(x), operand(x))
        return operable(f)
    return function_op
 
for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
    try:
        op(1,2)
    except TypeError:
        pass
    else:
        setattr(operable, name, op_to_function_op(op))
        
##
# The Lie Magic goes here!
def L_bracket(f,g):
    c = operable(jcb(f)) * g
    cinv = operable(jcb(g)) * f
    print(c(np.array([1.,1.,1.])))
    print(cinv(np.array([1.,1.,1.])))
#L_bracket(f,g)

def L_d(h,f,order=1):
    c = [h]
    for ii in range(order):
        c.append(np.dot(operable(egrad(c[ii])),f))
    
    return c[-1]


if __name__ == '__main__':
    @operable
    def f(x):
        return np.array([-x[1],-x[0],-x[2] + x[1]])

    @operable
    def g(x):
        #return np.sin(x + np.pi/2)
        return np.array([0.,1.,np.sin(x[2])])
    
    @operable
    def h(x):
        return 2*x[0] + 3*x[2]

    x_0 = np.array([1.,2.,5.])
    y_dot = L_d(h,f,order=2)
    print(np.sum(y_dot(x_0)))
