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

from mayavi import mlab

import pdb
import matplotlib.pyplot as plt

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

# this is so wrong; why am I taking a GRADIENT to find the time-domain fixed pt?
# FIX THIS
def f_points(f,args,epsilon=1e-1):
    
    if 'D' in args.keys():
        grad_f = np.sum(egrad(f)([args['x'],args['y'],args['z']],args['D']),axis=0).squeeze()
    else:
        grad_f = np.sum(egrad(f)([args['x'],args['y'],args['z']]),axis=0).squeeze().astype(np.float)
    
    
    plt.figure()
    plt.hist(grad_f.flatten(),bins=200)
    
    #pdb.set_trace()
    return (np.abs(grad_f) <= epsilon).astype(np.int)

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

def plot_Ldot(func,D=[],normed=True):
    x,y,z = gen_meshgrid(dens=20)
    
    
    if D == []:
        potgrid = func([x,y,z])
    else:
        potgrid = func([x,y,z,D])
        
        
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

def plot_field(dyn_field,coords,normfield=False):
    x = coords[0]
    y = coords[1]
    z = coords[2]
    #This plots the dynamics field first
    row_sums = dyn_field.sum(axis=0)
    if normfield:
        norm_dyn_field = 100* dyn_field / row_sums
    else:
        norm_dyn_field = dyn_field
    
    dyn_field[np.isinf(dyn_field)] = np.nan
    norm_dyn_field[np.isinf(norm_dyn_field)] = np.nan
    
    obj = quiver3d(x,y,z,norm_dyn_field[0,:,:,:],norm_dyn_field[1,:,:,:],norm_dyn_field[2,:,:,:])
    
def plot_fields(dyn_field,ctrl_field,coords,normfield=False):
    x = coords[0]
    y = coords[1]
    z = coords[2]
    #This plots the dynamics field first
    row_sums = dyn_field.sum(axis=0)
    if normfield:
        norm_dyn_field = 100* dyn_field / row_sums
    else:
        norm_dyn_field = dyn_field
    
    dyn_field[np.isinf(dyn_field)] = np.nan
    norm_dyn_field[np.isinf(norm_dyn_field)] = np.nan
    
    obj = quiver3d(x,y,z,norm_dyn_field[0,:,:,:],norm_dyn_field[1,:,:,:],norm_dyn_field[2,:,:,:])
    
    obj2 = quiver3d(x,y,z,ctrl_field[0,:,:,:],ctrl_field[1,:,:],ctrl_field[2,:,:],opacity=0.1)
    
    #plot_Ldot(y_dot)


if __name__=='__main__':
    plot_fields()
#%%
#Specific examples here