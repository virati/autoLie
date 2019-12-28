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

import mayavi
from mayavi.mlab import *
import sklearn.preprocessing as preproc
from dyn_lib import *

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
    cf = np.dot(operable(jcb(f)),g)
    cb = np.dot(operable(jcb(g)),f)
    #cf = L_d(f,g)
    #cb = L_d(g,f)
    #print(c(np.array([1.,1.,1.])))
    #print(cinv(np.array([1.,1.,1.])))
    return cf,cb

def L_d(h,f,order=1):
    c = [h]
    for ii in range(order):
        c.append(np.dot(operable(egrad(c[ii])),f))
    
    return c[-1]

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
    
    
    #pdb.set_trace()
    #return (np.abs(grad_f) <= epsilon).astype(np.int)

#%%
# Plotting methods here
# this here plots the potential grid arising from the function itself
def plot_LD(func,normalize=False,dens=10):
    x,y,z = gen_meshgrid(dens)
    
    potgrid = func([x,y,z])
    
    #plt.figure()
    #plt.pcolormesh(X,Y)
    
    if normalize:
        for xx in range(dens):
            for yy in range(dens):
                for zz in range(dens):
                    potgrid[:,xx,yy,zz] = potgrid[:,xx,yy,zz] / np.linalg.norm(potgrid[:,xx,yy,zz])
                    
            #potgrid[ii,:,:,:] = (potgrid[ii,:,:,:]) / (np.linalg.norm(potgrid[ii,:,:,:]))
            
    else:
        potgrid = potgrid / 500
        
    #pdb.set_trace()
    potgrid = 10*potgrid
    obj = quiver3d(x, y, z, potgrid[0,:,:,:], potgrid[1,:,:,:], potgrid[2,:,:,:], line_width=3, scale_factor=0.1,opacity=0.2,color=(0.0,1.0,0.0))

# here we find the lie for the Laplacian
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

def plot_field(dyn_field,coords,normfield=False,color=''):
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
    
    if color == '':
        obj = quiver3d(x,y,z,norm_dyn_field[0,:,:,:],norm_dyn_field[1,:,:,:],norm_dyn_field[2,:,:,:],opacity=0.8)
    else:
            
        obj = quiver3d(x,y,z,norm_dyn_field[0,:,:,:],norm_dyn_field[1,:,:,:],norm_dyn_field[2,:,:,:],opacity=0.8,color=color,mode='2dhooked_arrow')
    
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
    x_interest = np.array([1.0,-5.0,3.0]).T
    b1,b2 = L_bracket(f1,g1)
    print((b1(x_interest) - b2(x_interest)))
    
    #test = L_d(h1,f1)
    #print(test(x_interest))
    
    #plot_fields()
#%%
#Specific examples here