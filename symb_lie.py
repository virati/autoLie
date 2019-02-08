#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:41:12 2018

@author: virati
Symbolic differentiation layer for lie brackets
"""

import sympy as sym
from sympy.diffgeom import *
import numpy as np
from sympy import lambdify

M = Manifold("M",5)
P = Patch("P",M)
coord = CoordSystem("coord",P,["x","y","L","u","v"])

x,y,L,u,v = coord.coord_functions()

e_x, e_y, e_L, e_u,e_v = coord.base_vectors()

expr = (x+u)*e_x + (y+v)*e_y + L*e_L
print(LieDerivative(expr, sym.sqrt(L**2 + (y-x)**2)))

f = lambdify(x,LieDerivative(expr, sym.sqrt(L**2 + (y-x)**2)))


