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


def plot_LD(func):
    
    potgrid = func([x,y,z])
    
    #plt.figure()
    #plt.pcolormesh(X,Y)
    
    for ii in range(3):
        potgrid[ii,:,:,:] = (potgrid[ii,:,:,:]) / 500 #(np.linalg.norm(potgrid[ii,:,:,:]))
    
    obj = quiver3d(x, y, z, potgrid[0,:,:,:], potgrid[1,:,:,:], potgrid[2,:,:,:], line_width=3, scale_factor=1,opacity=0.2)

def plot_Ldot(func):
    x,y,z = gen_meshgrid(dens=20)
    
    potgrid = func([x,y,z])
    obj = points3d(x, y, z, np.sum(potgrid,axis=0),scale_factor=0.01,opacity=1)

    
#if __name__ == '__main__':
@operable
def f(x):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    #return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])
    #return np.array([-np.sin(x[1]), -5*x[0]**2, -np.sin(x[2] - x[1])])
    return - np.array([(x[0])*(x[0]-2)*(x[0]+2),x[1]**2,x[2]**2])
    #return -np.array([x[0]**2,x[1],x[2]])

@operable
def g(x):
    #return np.sin(x + np.pi/2)
    return np.array([np.sin(x[1]/10),np.sqrt(x[0]*x[1]),np.sin(x[2])])

@operable
def h(x):
    return 2*x[0] + 3*x[2]


def gen_meshgrid(dens=20):
    x_ = np.linspace(-10,10,dens)
    y_ = np.linspace(-10,10,dens)
    z_ = np.linspace(-10,10,dens)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    return x,y,z

def vector_example():
    y_dot = L_d(g,f,order=1)
    
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    
    dyn_field = f([x,y,z])
    ctrl_field = g([x,y,z])
    
    #This plots the dynamics field first
    obj = quiver3d(x,y,z,dyn_field[0,:,:,:],dyn_field[1,:,:,:],dyn_field[2,:,:,:])
    obj2 = quiver3d(x,y,z,ctrl_field[0,:,:,:],ctrl_field[1,:,:],ctrl_field[2,:,:])
    plot_Ldot(y_dot)

def scalar_example():
    x_0 = np.array([1.,2.,5.])
    y_dot = L_d(h,f,order=1)
    print(np.sum(y_dot(x_0)))
    
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    
    dyn_field = f([x,y,z])
    read_field = h([x,y,z])
    #This plots the dynamics field first
    obj = quiver3d(x,y,z,dyn_field[0,:,:,:],dyn_field[1,:,:,:],dyn_field[2,:,:,:])
    obj2 = points3d(x,y,z,read_field[:,:,:],colormap='copper',scale_factor=0.01)
    plot_LD(y_dot)