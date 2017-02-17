#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:08:18 2017

@author: coelhorp
"""

import numpy as np
import matplotlib.pyplot as plt
from diffusionmap import get_diffusionEmbedding

def distance_euc(x,y):
    return np.linalg.norm(x-y)

def gen_data(Npoints):

    state0 = np.array([+1, +1, +1])
    state1 = np.array([-1, -1, -1])
    sigma = 0.50
    state0_pertub = np.array([state0 + sigma*np.random.randn(3) for _ in range(Npoints/2)])
    state1_pertub = np.array([state1 + sigma*np.random.randn(3) for _ in range(Npoints/2)])
    points = np.concatenate((state0_pertub, state1_pertub), axis=0)    
    
    theta = np.zeros(Npoints)
    theta[Npoints/2:] = 1
    
    return points, theta

Npoints = 200
distance = distance_euc

points, theta = gen_data(Npoints)              
u, s = get_diffusionEmbedding(points, distance)

fig = plt.figure(figsize=(18,7))
colorst = [['b', 'r'][int(t)] for t in theta]  

ax = fig.add_subplot(121, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], color=colorst)
    
ax = fig.add_subplot(122)
ax.scatter(range(Npoints), u[:,1], color=colorst)






