#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:47:58 2018

@author: virati
Vineet Tiruvadi
virati@gmail.com


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

from dyn_lib import *
from op_plus import *

import pdb
npr.seed(1)
        
# The Lie Magic goes here!
def L_bracket(f,g,pt=[]):
    cf = np.dot(operable(jcb(f,0)),g)
    cb = np.dot(operable(jcb(g,0)),f)
    #cf = df_h(f,g)
    #cb = df_h(g,f)
    #print(c(np.array([1.,1.,1.])))
    #print(cinv(np.array([1.,1.,1.])))
    if pt == []:
        return cf,cb
    else:
        return np.sum(cf(pt) - cb(pt),axis=0)

def L_d(h,f,order=1):
    c = [h]
    for ii in range(order):
        c.append(np.dot(operable(egrad(c[ii],0)),f))
    
    return c[-1]


#%%
def L_dot(h,f,order=1):
    return np.sum(L_d(h,f,order=order))

# this is so wrong; why am I taking a GRADIENT to find the time-domain fixed pt?
# FIX THIS
def f_points(f,args,epsilon=1):
    
    if 'D' in args.keys():
        grad_f = (f([args['x'],args['y'],args['z']],args['D']))
    else:
        grad_f = (f([args['x'],args['y'],args['z']]))
    
    middle_gf = np.abs(grad_f) <= epsilon
    
    output = middle_gf.all(axis=0)
    
    #plt.figure()
    #plt.hist(np.abs(grad_f).flatten(),bins=20)
    
    #pdb.set_trace()
    return output


#%%
# Misc stuff here

if __name__=='__main__':
    x_interest = np.array([1.0,-2.0,3.0]).T
    b1,b2 = L_bracket(f1,g1)
    #print(b1(x_interest) - b2(x_interest))
    print(np.sum(b1(x_interest,0) - b2(x_interest,0),axis=1))
    #print((b1(x_interest)))
    
    #test = L_d(h1,f1)
    #print(test(x_interest))
    
    #plot_fields()
#%%
#Specific examples here