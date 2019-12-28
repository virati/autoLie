#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:42:49 2019

@author: virati
lie_plots
Library for plotting Lie analyses using Mayavi (predominantly)
"""
import matplotlib.pyplot as plt
import mayavi
from mayavi import mlab
from mayavi.mlab import *




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


