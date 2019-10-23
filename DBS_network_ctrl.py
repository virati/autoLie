#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:35:02 2019

@author: virati
The main file for the Lie-algebra, neural network project with Jirsa Lab
This is the big kahoona
"""

from lie_lib import *
import networkx as nx
import pdb

@operable
def f_hopf(p):
    #Node-based dynamics done here
    # c is a function of the network inputs into the node
    r_dot = p[0] * (p[0] - self.c)
    theta_dot = self.w * 1/(1-np.tanh(p[0]-self.c))
    
    return np.array([r_dot,theta_dot])

