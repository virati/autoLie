#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:35:02 2019

@author: virati
The main file for the Lie-algebra, neural network project with Jirsa Lab
This is the big kahoona
"""

import sys
sys.path.append('../src/')

from lie_lib import *
import networkx as nx
import ipdb
import autograd.numpy as np

''' First, we'll define the functions we're interested in '''
def integrator(f,state,params):
    dt=0.01
    k1 = f(state,params) * dt
    k2 = f(state + .5*k1,params)*dt
    k3 = f(state + .5*k2,params)*dt
    k4 = f(state + k3,params)*dt
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4)/6
    #new_state += np.random.normal(0,10,new_state.shape) * dt
    
    return new_state

class dyn_system:
    f_drift = []
    g_ctrl = []
    u = []
    
    def __init__(self):
        pass
        
    def simulate(self,state,tsteps=np.linspace(0,10,1000)):
        #f = self.f_drift + self.g_ctrl * self.u
        raster = []
        for ii,time in enumerate(tsteps):
            x_new = integrator(self.f_drift, state, self.P) + integrator(self.g_ctrl,state,self.P)
            raster.append(x_new)
            
        self.sim_raster = raster

class control_system(dyn_system):
    def __init__(self):
        #set our drift dynamics
        self.f_drift = f_kura
        self.g_ctrl = g_mono
        self.u = u_step
        self.h = h_single
        
        #set our graph
        n_elements = 10
        n_regions = int(np.floor(n_elements/2))
        self.G = nx.random_regular_graph(4, n_elements)
        self.L = nx.linalg.laplacian_matrix(self.G).todense()
        self.D = nx.linalg.incidence_matrix(self.G).todense()
        # for each of our elements, assign them to a brain region
        self.e_to_r = np.random.randint(0,n_regions,size=n_elements)
        
        #do our disease layer
        n_symp = 2
        #self.Xi = np.random.randint(0,1,size=(n_regions,n_symp))
        self.Xi = Xi
        
        self.P = self.L
        
        self.x_state = np.random.uniform(size=(1000,1))
        
        self.n_regions = n_regions
        self.n_symp = n_symp
        self.n_elements = n_elements
    
    def disease_measure(self):
        self.interact_vector = L_d(self.Xi,self.h)
        rand_checks = np.random.uniform(-10,10,size=(self.n_elements,self.n_elements))
        is_zero = []
        measure_sets = []
        for ii in range(self.n_elements):
            measure_sets.append(self.interact_vector(rand_checks[ii,:].reshape(1,-1),[1]))
            is_zero.append(measure_sets[-1] == 0)
            #print(measure_set)
        
        #pdb.set_trace()
        print('Measurement-Disease interaction: ' + str(np.array(is_zero).all() == True))
        
                                   
    ''' The main result from our ability to control the disease state through g'''
    def disease_control(self):
        self.interact_vector = L_d(self.Xi,self.g_ctrl)
        #choose N random vectors in the N dim space
        rand_checks = np.random.uniform(-10,10,size=(self.n_elements,self.n_elements))
        is_zero = []
        for ii in range(self.n_elements):
            control_set = self.interact_vector(rand_checks[ii,:].reshape(1,-1),[1])
            is_zero.append(control_set == 0)
            #print(control_set)
        
        print('Control-Disease interaction: ' + str(np.array(is_zero).all() == True))
        #need to check if this is zero *everywhere*
        #We'll multiply the above by the x vector, then check if the gradient is zero
        #grid = gen_arbgrid(dims=5)
        
        
        #controlset = self.interact_vector(grid,[1])
        
        #return controlset == nullset
    
    ''' Below isn't necessary for ASSFN project'''
    def disease_bracket(self):
        #ipdb.set_trace()
        b1,b2 = L_bracket(f_kura,Xi)
        x_state = self.x_state
        
        test = op_to_function_op(b1) #THIS is where the problem is arising, when we're actually COMPUTING, it doesn't focus just on the first argument
        
        print(test)
        
    def laplac(self):
        return nx.linalg.laplacian_matrix(self.G).todense()

def Xi(x,P):
    return np.dot(np.array([0.0,1.0,0.0,-0.0,0.,0.,0.,0.,0.0,0.0]).reshape(1,-1),x.T)
    #return np.dot(np.random.randint(0,1,size=(10,P[0])),x)

# We first care about the drift dynamics
#@operable
def f_hopf(x,P):
    r = x[:,0].reshape(-1,1)
    theta = x[:,1].reshape(-1,1)
    
    #Node-based dynamics done here
    # c is a function of the network inputs into the node
    neighbors = np.dot(L,r)
    
    r_dot = np.diag(np.outer(r,r - neighbors)).reshape(-1,1)
    
    #theta_dot = self.w * 1/(1-np.tanh(p[0]-self.c))
    theta_dot = 0.02 * np.exp(r)
    
    
    return np.array([r_dot,theta_dot])

def f_consensus(x,P):
    x_dot = -np.dot(P[0],x)
    
    return x_dot

def f_kura(x,P):
    D = P
    x_1 = np.dot(D.T,x)
    x_2 = np.sin(x_1)
    x_3 = np.dot(D,x_2)
    
    return x_3
    
def g_mono(x,P):
    ret_vec = np.zeros_like(x)
    ret_vec[0,0] = -x[0,0]
    ret_vec[0,8] = -x[0,2]
    #ret_vec[0,8] = x[0,1] - x[0,5]**2
    return ret_vec

def h_single(x,P):
    measure_vect = np.zeros_like(x)
    measure_vect[0,7] = 1.0
    return np.dot(measure_vect.T,x)

def u_step(t,P):
    return (t > 5).astype(np.float)

#%% Script running code
    
if __name__=='__main__':
    
    brain = control_system()
    #brain.simulate(state=x0)
    brain.disease_control()
    brain.disease_measure()
