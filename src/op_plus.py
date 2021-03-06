#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:39:50 2019

@author: virati
Operator plus library
"""
import operator

class operable:
    def __init__(self, f):
        self.f = f
    def __call__(self,x,args):
        return self.f(x,args)
 
# Convert operators to the corresponding function
def op_to_function_op(op):
    def function_op(self, operand):
        def f(x,args):
            return op(self(x,args), operand(x,args))
        return operable(f)
    return function_op
 
for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
    try:
        op(1,2)
    except TypeError:
        pass
    else:
        setattr(operable, name, op_to_function_op(op))
       

