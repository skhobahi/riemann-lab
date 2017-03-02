#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:36:27 2017

@author: coelhorp
"""

import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann
from mpl_toolkits.mplot3d import Axes3D

filepath = '/localdata/coelhorp/sleepdata/cov_mtrs_ins3_2sec.npy'
(cov_mtrs,stages) = np.load(filepath)

ntrials = 125
X = []
y = []
for i,stage in enumerate(['S3', 'RM','S1','S2']):
    Xi = cov_mtrs[stages[stage]][:ntrials]
    X.append(Xi)
    y = y + [i for _ in range(len(Xi))]
covs = np.concatenate(X, axis=0)

#%%

from diffusionmap import get_diffusionEmbedding
u,l = get_diffusionEmbedding(points=covs, distance=distance_riemann)

#%%

fig = plt.figure(figsize=(11, 9.6))
ax  = fig.add_subplot(111, projection='3d')
colors = ['r','g','b','b']
for i in range(len(covs)):
    ax.scatter(u[i,1], u[i,2], u[i,3], color=colors[y[i]], s=44)
      
