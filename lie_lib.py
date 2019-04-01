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

from mayavi.mlab import *
import sklearn.preprocessing as preproc


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

def L_dot(h,f,order=1):
    return np.sum(L_d(h,f,order=order))


#%%
# Plotting methods here

def plot_LD(func):
    x,y,z = gen_meshgrid(dens=20)
    
    potgrid = func([x,y,z])
    
    #plt.figure()
    #plt.pcolormesh(X,Y)
    
    for ii in range(3):
        potgrid[ii,:,:,:] = (potgrid[ii,:,:,:]) / 500 #(np.linalg.norm(potgrid[ii,:,:,:]))
    
    obj = quiver3d(x, y, z, potgrid[0,:,:,:], potgrid[1,:,:,:], potgrid[2,:,:,:], line_width=3, scale_factor=1,opacity=0.2)

def plot_Ldot(func,normed=True):
    x,y,z = gen_meshgrid(dens=20)
    
    potgrid = func([x,y,z])
    if normed:
        obj = points3d(x, y, z, np.abs(np.log10(np.sum(potgrid,axis=0))),scale_factor=1,opacity=1)
    else:
        obj = points3d(x, y, z, (np.sum(potgrid,axis=0)),scale_factor=0.01,opacity=1)



#%%
# Misc stuff here


def gen_meshgrid(dens=20):
    x_ = np.linspace(-10,10,dens)
    y_ = np.linspace(-10,10,dens)
    z_ = np.linspace(-10,10,dens)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    return x,y,z
    
def plot_fields(dyn_field,ctrl_field,coords):
    x = coords[0]
    y = coords[1]
    z = coords[2]
    #This plots the dynamics field first
    row_sums = dyn_field.sum(axis=0)
    norm_dyn_field = dyn_field / row_sums
    
    obj = quiver3d(x,y,z,norm_dyn_field[0,:,:,:],dyn_field[1,:,:,:],dyn_field[2,:,:,:])
    obj2 = quiver3d(x,y,z,ctrl_field[0,:,:,:],ctrl_field[1,:,:],ctrl_field[2,:,:])
    
    #plot_Ldot(y_dot)


#%%
#Specific examples here