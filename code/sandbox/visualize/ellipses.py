#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:43:24 2017

@author: coelhorp
"""

'''
xy:     center of ellipse
width:  length of horizontal axis
height: length of vertical axis
angle:  rotation in degrees (anti-clockwise)
theta1: starting angle of the arc in degrees
theta2: ending angle of the arc in degrees
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from numpy import cos, sin

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.mean import mean_euclid

def gen_spd(n = 2):
    C = np.random.randn(n,n)
    C = C+C.T
    _,v = np.linalg.eig(C)
    w = 5*np.random.rand(n)
    w = np.diag(w)
    return np.dot(v, np.dot(w, v.T))

def gen_ellipse(Q, center=[0,0], lw=1.0, ls='-', fc='none', ec='black'): 
    w,v = np.linalg.eig(Q)
    height = 2.0/w[0]
    width  = 2.0/w[1] 
    theta  = np.arccos(v[0,0])/(2*np.pi)*360    
    e = Ellipse(xy=center,
                width=width,
                height=height,
                angle=theta,
                lw=lw,
                ls=ls,
                fc=fc,
                ec=ec)    
    return e    
    
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, aspect='equal')

theta = 0
Q = np.array([[cos(theta), -sin(theta)],
              [sin(theta),  cos(theta)]])    
D = np.eye(2)
D[0,0] = 3.0
D[1,1] = 1.0
C = np.dot(Q, np.dot(D, Q.T)) 

theta = np.pi/2.0
R = np.array([[cos(theta), -sin(theta)],
               [sin(theta),  cos(theta)]])
Coutlier = np.dot(R,np.dot(C, R.T)) 

covs = [C, Coutlier]
for _ in range(5):
    theta = 1*np.pi/20*np.random.randn()
    R  = np.array([[cos(theta), -sin(theta)],
                   [sin(theta),  cos(theta)]])
    Cr = np.dot(R,np.dot(C, R.T))
    covs.append(Cr)    
    
for cov in covs:
    e = gen_ellipse(cov)
    ax.add_artist(e)   

covs = np.stack(covs)    
Crmn = mean_riemann(covs)
Ceuc = mean_euclid(covs)

e = gen_ellipse(Crmn, ec='blue', lw=2.0)
ax.add_artist(e)
e = gen_ellipse(Ceuc, ec='red', lw=2.0)
ax.add_artist(e)    
    
ax.set_xlim(-1.2, +1.2)
ax.set_ylim(-1.2, +1.2)

plt.show()








