#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 08:40:29 2019

@author: virati
"""

#%%
# Our functions of interest HERE

from lie_lib import *
    
#if __name__ == '__main__':
@operable
def f(x):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    #return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])
    #return np.array([-np.sin(x[1]), -5*x[0]**2, -np.sin(x[2] - x[1])])
    #return - np.array([(x[0])*(x[0]-2)*(x[0]+2),x[1]**2,x[2]**2])
    #return -np.array([x[0],x[1],x[2]])
    return -np.array([np.sin(x[2]),np.sin(x[0]),np.sin(x[1])])
    #return -np.array([x[0] + x[1], x[1]*x[2],x[2]**2])

@operable
def g(x):
    #return np.sin(x + np.pi/2)
    return np.array([np.sin(x[1]/10),np.cos(x[0]*x[1]),np.sin(x[2])])

@operable
def h(x):
    return 2*x[0] + 3*x[2]

def vector_example():
    y_dot = L_d(g,f,order=1)
    
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    coords = (x,y,z)
    
    dyn_field = f([x,y,z])
    ctrl_field = g([x,y,z])
    
    plot_fields(dyn_field,ctrl_field,coords)
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


#%%
    
if __name__ == '__main__':
    scalar_example()