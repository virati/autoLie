#!/usr/bin/env python3
"""
Operator plus library

@author: virati
Vineet Tiruvadi
virati@gmail.com
"""
import operator


class operable:
    """Wrapper class that makes functions operable with standard operators."""

    def __init__(self, f):
        self.f = f

    def __call__(self, x, args):
        return self.f(x, args)


# Convert operators to the corresponding function
def op_to_function_op(op):
    """Convert Python operators to function operations."""
    def function_op(self, operand):
        def f(x, args):
            return op(self(x, args), operand(x, args))
        return operable(f)
    return function_op


for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
    try:
        op(1, 2)
    except TypeError:
        pass
    else:
        setattr(operable, name, op_to_function_op(op))
       

