#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:30:32 2019

@author: virati
An example of 
"""

# needs mayavi2
# run with ipython -wthread
import networkx as nx
import numpy as np
from mayavi import mlab
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# some graphs to try
#H=nx.krackhardt_kite_graph()
#H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
#H=nx.grid_2d_graph(4,5)
#H=nx.cycle_graph(20)
H = nx.erdos_renyi_graph(20,0.3)

L = np.abs(nx.laplacian_matrix(H)).todense()

theta = np.random.multivariate_normal(12*np.ones(L.shape[0]),L,10000).T
#%%
#plt.plot(y)

t = np.linspace(0,10,10000)
y = np.zeros((16,10000))
for ii in range(16):
    y[ii,:] = np.sin(2 * np.pi * theta[ii,:] * t)

plt.plot(y[:,0:1000].T)

#%%
# Now do empiric variance
empir_var = np.zeros((16,16))
for ii in range(16):
    for jj in range(16):
        empir_var[ii,jj] = np.cov(np.vstack((y[ii,:],y[jj,:])))[0,1]


plt.figure()
plt.subplot(211)
plt.title('Estimated Laplacian through Covar')
plt.imshow(empir_var > 0.01)
plt.subplot(212)
plt.imshow(L)
plt.title('True Laplacian')

#%%
def plot_graph():
    G = nx.erdos_renyi_graph(7, 0.2)
    plt.figure()
    nx.draw(G)

plot_graph()
#%%
    
def render_graph(H,read,write):
    # reorder nodes from 0,len(G)-1
    G=nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos=nx.spring_layout(G,dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars=np.array(G.nodes())+5
    
    mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.clf()
    
    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        scalars,
                        scale_factor=0.1,
                        scale_mode='none',
                        colormap='Blues',
                        resolution=20)
    pts_reads = mlab.points3d(xyz[read,0], xyz[read,1], xyz[read,2],
                    scalars[read],
                    scale_factor=0.1,
                    scale_mode='none',
                    color=(0.0,1.0,0.0),
                    resolution=20)
    pts_reads = mlab.points3d(xyz[write,0], xyz[write,1], xyz[write,2],
                scalars[write],
                scale_factor=0.1,
                scale_mode='none',
                color=(1.0,0.0,0.0),
                resolution=20)
    
    
    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8),opacity=0.1)
    
    mlab.savefig('mayavi2_spring.png')
    #mlab.show() # interactive window
    
read = np.zeros((20,)).astype(np.bool)
write = np.zeros((20,)).astype(np.bool)
read[2] = True
write[5] = True
render_graph(H,read,write)