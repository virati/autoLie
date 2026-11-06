#!/usr/bin/env python3
"""
Lie Derivatives on predefined operators

@author: virati
Vineet Tiruvadi
virati@gmail.com
"""

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import grad
from autograd import elementwise_grad as egrad
from autograd import jacobian as jcb
from functools import reduce

from autolie.dyn_lib import *
from autolie.op_plus import *

import pdb

npr.seed(1)


# The Lie Magic goes here!
def L_bracket(f, g, args=[]):
    cf = np.dot(operable(jcb(f, 0)), g)
    cb = np.dot(operable(jcb(g, 0)), f)
    # cf = df_h(f,g)
    # cb = df_h(g,f)
    # print(c(np.array([1.,1.,1.])))
    # print(cinv(np.array([1.,1.,1.])))
    if args == []:
        return cf, cb
    else:
        # The products above are broadcast elementwise, J[i, j] * v[j]; the
        # matrix-vector product is the sum over j. Summing over axis 0 (as this
        # used to) gives sum_i J[i, j] v[j], which is not a bracket.
        return np.sum(cf(args[0], args[1]) - cb(args[0], args[1]), axis=1)


def bracket_field(f, g):
    """The vector field [f, g](x) = (df/dx) g(x) - (dg/dx) f(x), as an operable.

    f and g take (x, args) and return arrays shaped like x (a single point).
    This is L_bracket's convention; the control texts that write
    [f, g] = (dg/dx) f - (df/dx) g get the negative, i.e. bracket_field(g, f).
    """
    Jf, Jg = jcb(f, 0), jcb(g, 0)
    return operable(lambda x, args: np.dot(Jf(x, args), g(x, args)) - np.dot(Jg(x, args), f(x, args)))


def _lie_once(h, f):
    Jh = jcb(h, 0)
    return operable(lambda x, args: np.dot(Jh(x, args), f(x, args)))


def L_d(d, f, order=1):
    """The Lie derivative L_f^order d: the rate of change of d along the flow of f.

    For a scalar d this is grad(d) . f; for a vector-valued d it is (dd/dx) f,
    taken component by component. Higher orders apply it repeatedly, so
    L_d(h, f, 2) is grad(L_f h) . f.

    This used to multiply grad(d) by f elementwise and never sum, so a scalar
    Lie derivative came back as the vector of its terms; and since autograd's
    elementwise_grad differentiates the sum of a function's outputs, a vector
    d got the gradient of its component sum in place of its Jacobian.
    """
    for _ in range(order):
        d = _lie_once(d, f)
    return d


# %% UNCLEAR USAGE BELOW
def L_dot(h, f, order=1):
    # np.sum of an operable is the operable itself, so sum inside the call.
    Lh = L_d(h, f, order=order)
    return operable(lambda x, args: np.sum(Lh(x, args)))


# this is so wrong; why am I taking a GRADIENT to find the time-domain fixed pt?
# FIX THIS
def f_points(f, args, epsilon=1):
    if "D" in args.keys():
        grad_f = f([args["x"], args["y"], args["z"]], args["D"])
    else:
        grad_f = f([args["x"], args["y"], args["z"]])

    middle_gf = np.abs(grad_f) <= epsilon

    output = middle_gf.all(axis=0)

    # plt.figure()
    # plt.hist(np.abs(grad_f).flatten(),bins=20)

    # pdb.set_trace()
    return output


# %% Unit test
if __name__ == "__main__":
    x_interest = np.array([1.0, -2.0, 3.0]).T
    test = L_d(h1, f1)
    print(test(x_interest, 0))

    b1, b2 = L_bracket(f1, g1)
    print(np.sum(b1(x_interest, 0) - b2(x_interest, 0), axis=1))
# %%
# Specific examples here
