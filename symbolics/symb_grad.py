#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:41:32 2018

@author: virati
Symbolic gradient
"""


from sympy import Symbol, Matrix, Function, simplify

x = Symbol('x')




#%%



eta = Symbol('eta')
xi = Symbol('xi')

x = Matrix([[xi],[eta]])

h = [Function('h_'+str(i+1))(x[0],x[1]) for i in range(3)]
z = [Symbol('z_'+str(i+1)) for i in range(3)]

lamb = 0
sigma = 1
for i in range(3):
    lamb += 1/(2*sigma**2)*(z[i]-h[i])**2
simplify(lamb)

lgrad = [lamb.diff(x) for x in z+[eta]]