#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:54:19 2017

@author: coelhorp
"""

import sys
sys.path.append('../')
import glob

import numpy as np
import scipy as sp
from scipy.signal import welch
from scipy import signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colour import Color

from pyriemann.utils.distance import distance_riemann
from utilities.diffusion_map import get_diffusionEmbedding
from pyriemann.estimation import Covariances

path  = '/localdata/coelhorp/epilepsy/seizure_detection/Patient_8'
filepaths = sorted(glob.glob(path + '/*_ictal_*'))

X = []
lat = []
for filepath in filepaths:
    struct = sp.io.loadmat(filepath)
    X.append(struct['data'])  
    lat.append(struct['latency'][0])
    
lat = np.array(lat)    
X = np.stack(X)    

#%%

fs = struct['freq']
fini = 1.0
fend = 40.0
b,a = signal.butter(5, [fini/(fs/2), fend/(fs/2)], btype='bandpass')
for xt in X:
    f,pxx = welch(xt, fs=fs)
    xt = signal.filtfilt(b,a,xt)

#%%

covs = Covariances(estimator='oas').fit_transform(X)
print 'getting the diffusion embedding'
u,l = get_diffusionEmbedding(points=covs, distance=distance_riemann)

#%%

fig    = plt.figure(figsize=(11, 9.6))
ax     = fig.add_subplot(111, projection='3d')
green  = Color("green")

colors = list(green.range_to(Color("red"), int(np.max(lat)+1)))
for i in range(len(covs)):
    ax.scatter(u[i,1], u[i,2], u[i,3], color=colors[int(lat[i])].get_rgb(), s=44)
ax.set_xlim(-3,+3)
ax.set_ylim(-3,+3)
ax.set_zlim(-3,+3)







