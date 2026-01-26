#!/usr/bin/env python3
"""
LIE ANALYSIS of drift and control dynamics

This script demonstrates Lie analysis of drift and control dynamics
using various functions for dynamics/control and plotting.

@author: virati
Vineet Tiruvadi
virati@gmail.com
"""

import sys
sys.path.append('../src')

from lie_lib import *
import networkx as nx
import pdb
import pyvista as pv
import lie_plots as lip

#if __name__ == '__main__':
@operable
def f1(x):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])

@operable
def f2(x):
    return np.array([-np.sin(x[1]), -5*x[0]**2, -np.sin(x[2] - x[1])])

@operable
def f3(x):
    return -np.array([(x[0])*(x[0]-2)*(x[0]+2),x[1]**2,x[2]**2])

@operable
def f4(x):
    return -np.array([x[0],x[1],x[2]])
@operable
def f5(x):
    return -np.array([np.sin(x[2]),np.sin(x[0]),np.sin(x[1])])
    #return -np.array([x[0] + x[1], x[1]*x[2],x[2]**2])
@operable
def f6(x):
    return -np.array([x[0],x[1],x[2]])

@operable
def f7(x,args):
    return -np.array([x[0]**3 - 4*x[0]**2,x[2]**3 - 2*x[1]**2,x[1]**3 - 3*x[2]])

@operable
def f8(x,args):
    return -np.array([x[1] * x[2],x[2] * (1.0 + x[1] - x[2]),x[1] * (-1.0 + x[2] + x[1])])

@operable
def f9(x,args):
    return -np.array([1.0*x[2],x[0],-1.0 * x[1]**2])

class f_net:
    def __init__(self,D):
        self.D = D
        
@operable        
def net_dyn(x,D):
    new_x = np.swapaxes(np.array(x),0,2)
    D = np.array(D)
    a1 = np.dot(D.T,new_x)
    a2 = np.swapaxes(np.sin(a1),0,2)
    a3 = np.dot(D,a2)
    
    return a3

@operable
def g(x):
    #return np.sin(x + np.pi/2)
    #return np.array([np.sin(x[1]/10),np.cos(x[0]*x[1]),np.sin(x[2])])
    return np.array([x[1],0*x[1],0*x[1]])

@operable
def h(x):
    return 2*x[0] + 3*x[2]


def network_dynamics_example():
    """Example of network dynamics visualization with fixed points."""
    use_func = f7
    x_ = np.linspace(-10, 10, 20)
    y_ = np.linspace(-10, 10, 20)
    z_ = np.linspace(-10, 10, 20)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    coords = (x, y, z)

    dyn_field = use_func([x, y, z])
    plotter = lip.plot_field(dyn_field, coords, normfield=False)

    args = {'x': x, 'y': y, 'z': z}
    zeros = f_points(use_func, args, epsilon=0.5)

    # Add fixed points as spheres
    if np.any(zeros):
        fps_points = np.column_stack([
            x[zeros].flatten(),
            y[zeros].flatten(),
            z[zeros].flatten()
        ])
        fps_cloud = pv.PolyData(fps_points)
        fps_spheres = fps_cloud.glyph(geom=pv.Sphere(radius=0.2), scale=False)
        plotter.add_mesh(fps_spheres, cmap='seismic')

    return plotter

def simple_control_example():
    """Example showing drift and control fields with Lie derivative."""
    drift = f7
    control = f9

    # Create plotter with black background
    plotter = pv.Plotter()
    plotter.set_background('black')

    x_ = np.linspace(-10, 10, 10)
    y_ = np.linspace(-10, 10, 10)
    z_ = np.linspace(-10, 10, 10)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    coords = (x, y, z)

    # Plot drift field in red
    dyn_field = drift([x, y, z], [0])
    plotter = lip.plot_field(dyn_field, coords, normfield=False, color=(1.0, 0.0, 0.0), plotter=plotter)

    # Plot control field in blue
    ctrl_field = control([x, y, z], [0])
    plotter = lip.plot_field(ctrl_field, coords, normfield=False, color=(0.0, 0.0, 1.0), plotter=plotter)

    # We do a Lie dot product
    y_dot = L_dot(control, drift, order=1)
    plotter = lip.plot_LD(y_dot, normalize=True, plotter=plotter)

    return plotter
    

def simple_dynamics_example():
    """Simple dynamics field visualization example."""
    use_func = f8
    x_ = np.linspace(-10, 10, 20)
    y_ = np.linspace(-10, 10, 20)
    z_ = np.linspace(-10, 10, 20)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    coords = (x, y, z)

    dyn_field = use_func([x, y, z])
    plotter = lip.plot_field(dyn_field, coords, normfield=False)

    args = {'x': x, 'y': y, 'z': z}
    zeros = f_points(use_func, args, epsilon=2)

    # Add fixed points
    if np.any(zeros):
        fps_points = np.column_stack([
            x[zeros].flatten(),
            y[zeros].flatten(),
            z[zeros].flatten()
        ])
        fps_cloud = pv.PolyData(fps_points)
        fps_spheres = fps_cloud.glyph(geom=pv.Sphere(radius=0.2), scale=False)
        plotter.add_mesh(fps_spheres, cmap='seismic')

    return plotter

def simple_trajectory():
    """Visualize trajectory through phase space."""
    x0 = np.array([0.0, 0.0, 0.0])
    usefunc = f8

    x_next = [np.copy(x0)]
    for ii in range(1000):
        x_next.append(1/1000000*usefunc(x_next[-1]))

    x_next = np.array(x_next)

    # Create trajectory as polyline
    plotter = pv.Plotter()
    points = x_next
    poly = pv.PolyData(points)
    poly.lines = np.hstack([[2, i, i+1] for i in range(len(points)-1)])

    tube = poly.tube(radius=0.02)
    plotter.add_mesh(tube)

    return plotter
    
def vector_example():
    """Example showing vector fields and Lie derivative."""
    f = f8
    g = f4
    y_dot = L_d(g, f, order=1)

    x_ = np.linspace(-10, 10, 20)
    y_ = np.linspace(-10, 10, 20)
    z_ = np.linspace(-10, 10, 20)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    coords = (x, y, z)

    dyn_field = f([x, y, z])
    ctrl_field = g([x, y, z])

    plotter = lip.plot_fields(dyn_field, ctrl_field, coords, normfield=True)
    plotter = lip.plot_Ldot(y_dot, plotter=plotter)

    return plotter

def scalar_example():
    """Example with scalar field visualization."""
    x_0 = np.array([1., 2., 5.])
    y_dot = L_d(h, f, order=1)
    print(np.sum(y_dot(x_0)))

    x_ = np.linspace(-10, 10, 20)
    y_ = np.linspace(-10, 10, 20)
    z_ = np.linspace(-10, 10, 20)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    plotter = pv.Plotter()

    # Plot dynamics field as arrows
    dyn_field = f([x, y, z])
    coords = (x, y, z)
    plotter = lip.plot_field(dyn_field, coords, plotter=plotter)

    # Plot scalar read field as point cloud
    read_field = h([x, y, z])
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    cloud = pv.PolyData(points)
    cloud['scalars'] = read_field.flatten()
    plotter.add_mesh(cloud, cmap='seismic', point_size=5.0, render_points_as_spheres=True)

    return plotter



if __name__ == '__main__':
    # EXAMPLE 1
    # plotter = simple_control_example()
    # plotter = simple_trajectory()

    # EXAMPLE 2
    plotter = simple_control_example()
    # plotter = vector_example()

    # Show the interactive window
    plotter.show()