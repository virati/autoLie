#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:41:13 2018

@author: virati
from: https://github.com/HIPS/autograd/issues/250
"""

import autograd.numpy as np
from autograd import grad
from autograd import jacobian as jcb

def sphere_potential(xy):
    R = 0.5
    r = np.sqrt(np.sum(xy**2, axis=-1))
    return np.minimum(1/r, 1/R)

def potential(xy):
    sphere_pos = np.array([0., 1.])
    return (  sphere_potential(xy - sphere_pos)
            - sphere_potential(xy + sphere_pos))

X, Y = np.meshgrid(*[np.linspace(-2, 2, 50)]*2)
xy_grid = np.stack([X, Y], axis=-1)

potential_grid = potential(xy_grid)
E_field_grid = -jcb(potential)(xy_grid)  # <--- evaluating the electric field

import matplotlib.pyplot as plt
plt.figure()
plt.pcolormesh(X, Y, potential_grid)
plt.quiver(X, Y, E_field_grid[..., 0], E_field_grid[..., 1])