#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:19:17 2019

@author: virati
Interactive gui for Lie fields
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD Style.

import numpy as np

from numpy import arange, pi, cos, sin

from traits.api import HasTraits, Range, Instance, \
        on_trait_change
from traitsui.api import View, Item, Group

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel

from dyn_lib import *
from lie_lib import *


class DEPRFieldModel(HasTraits):
    n_meridional    = Range(0, 30, 6, )#mode='spinner')
    n_longitudinal  = Range(0, 30, 11, )#mode='spinner')

    scene = Instance(MlabSceneModel, ())

    plot = Instance(PipelineBase)

    def __init__(self):
        self.drift = f3
        self.control = f6

        self.x_ = np.linspace(-10,10,20)
        self.y_ = np.linspace(-10,10,20)
        self.z_ = np.linspace(-10,10,20)
        
        self.x,self.y,self.z = np.meshgrid(self.x_,self.y_,self.z_,indexing='ij')

    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    @on_trait_change('n_meridional,n_longitudinal,scene.activated')
    def update_plot(self):
        #x, y, z, t = curve(self.n_meridional, self.n_longitudinal)
        drift_field = self.drift([self,x,self.y,self.z])
        print('test')
        
        if self.plot is None:
            self.plot = self.scene.mlab.quiver3d(self.x,self.y,self.z,dyn_field[0,:,:,:],dyn_field[1,:,:,:],dyn_field[2,:,:,:],opacity=0.5,color=color,mode='2dhooked_arrow')
        else:
            #self.plot.mlab_source.trait_set(x=x, y=y, z=z, scalars=t)
            self.plot.mlab_source.trait_set(x=self.x,y=self.y,z=self.z,u=drift_field[0,:,:,:],v=drift_field[1,:,:,:],w=drift_field[2,:,:,:])
            


    # The layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Group(
                        '_', 'n_meridional', 'n_longitudinal',
                     ),
                resizable=True,
                )
                
dphi = pi/1000.
phi = arange(0.0, 2*pi + 0.5*dphi, dphi, 'd')

def curve(f,c, d):
    x = np.linspace(-10,10,20)
    y = np.linspace(-10,10,20)
    z = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x,y,z,indexing='ij')
    field = f([x,y,z],bifur=(c,d))
    
    #t = sin(mu)
    return x, y, z, field

def compute_path(f,bifur,x0=(1.,1.,1.)):
    x_roster = [x0]
    for ii in range(1000):
        x_new = integrator(f,x_roster[ii],bifur)
        x_roster.append(x_new)
    
    return x_roster

def integrator(f,state,bifur):
    dt=0.01
    k1 = f(state,bifur=bifur) * dt
    k2 = f(state + .5*k1,bifur=bifur)*dt
    k3 = f(state + .5*k2,bifur=bifur)*dt
    k4 = f(state + k3,bifur=bifur)*dt
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4)/6
    #new_state += np.random.normal(0,10,new_state.shape) * dt
    
    return new_state

class MyModel(HasTraits):
    n_meridional    = Range(0, 30, 6, )#mode='spinner')
    n_longitudinal  = Range(0, 30, 11, )#mode='spinner')
    n_opac = Range(0.0,1.0,0.5,)

    scene = Instance(MlabSceneModel, ())

    plot = Instance(PipelineBase)
    plot2 = Instance(PipelineBase)
    use_func=f8

    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    @on_trait_change('n_meridional,n_longitudinal,n_opac,scene.activated')
    def update_plot(self):
        x, y, z, dyn = curve(f8,self.n_meridional, self.n_longitudinal)
        opac = self.n_opac
        #Recompute our trajectory
        traj = np.array(compute_path(f8,(self.n_meridional, self.n_longitudinal)))
        #print(opac)
        
        #pdb.set_trace()
        if self.plot is None:
            self.plot = self.scene.mlab.quiver3d(x, y, z, dyn[0,:,:,:],dyn[1,:,:,:],dyn[2,:,:,:])
            #self.plot2 = self.scene.mlab.plot3d(traj[:,0],traj[:,1],traj[:,2])#,scale_factor=0.01)
            self.plot2 = self.scene.mlab.points3d(traj[:,0],traj[:,1],traj[:,2],mode='2darrow')#,scale_factor=0.01)

        else:
            self.plot.mlab_source.trait_set(x=x, y=y, z=z, u=dyn[0,:,:,:],v=dyn[1,:,:,:],w=dyn[2,:,:,:])
            self.plot2.mlab_source.trait_set(x=traj[:,0],y=traj[:,1],z=traj[:,2])

        
        
    # The layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Group(
                        '_', 'n_meridional', 'n_longitudinal','n_opac',
                     ),
                resizable=True,
                )

my_model = MyModel()
my_model.configure_traits()
