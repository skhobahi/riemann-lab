#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:04:31 2017

@author: coelhorp

First steps with Diffusion Maps
Reproduced one of the figures from the article
Coifman and Lafon, "Diffusion maps" (2006)

"""

import numpy as np
from numpy import sin, cos

import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from diffusionmap import get_diffusionEmbedding

def distance_euc(x,y):
    return np.linalg.norm(x-y)

def gen_data(Npoints):
    points = np.zeros((Npoints, 3))
    p = 3
    q = 15
    r = 0.9
    theta = np.linspace(0, 2*np.pi/3, Npoints)
    points[:,0] = +r*cos(p*theta)
    points[:,1] = +r*sin(p*theta)
    points[:,2] = -r*sin(q*theta)
    
    idx = np.array(range(Npoints))
    np.random.shuffle(idx)
    points = points[idx]
    theta  = theta[idx]
    
    return points, theta

def get_color(x):
    # note that x goes from 0 to 1
    colormap = plt.cm.gist_rainbow
    return colormap(x)

Npoints = 150
distance = distance_euc

points, theta = gen_data(Npoints)              

fig = plt.figure(figsize=(16,7))

ax = fig.add_subplot(121, projection='3d')
for i,point in enumerate(points):
    thetan = theta[i]/max(theta)
    ax.scatter(point[0], point[1], point[2], color=get_color(thetan))
    
ax = fig.add_subplot(122)
u,s = get_diffusionEmbedding(points, distance, alpha=1.0)
for i in range(Npoints):
    thetan = theta[i]/max(theta)
    ax.scatter(s[1]*u[i,1], s[2]*u[i,2], color=get_color(thetan))
ax.axis('equal')
