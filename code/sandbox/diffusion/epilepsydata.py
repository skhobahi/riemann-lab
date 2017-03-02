#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:54:19 2017

@author: coelhorp
"""

import numpy as np
import scipy as sp
from scipy.signal import welch
from scipy import signal
import glob
from pyriemann.utils.distance import distance_riemann
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colour import Color

path  = '/localdata/coelhorp/epilepsy/seizure_detection/Patient_3'
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
fini = 4.0
fend = 40.0
b,a = signal.butter(5, [fini/(fs/2), fend/(fs/2)], btype='bandpass')
for xt in X:
    f,pxx = welch(xt, fs=fs)
    xt = signal.filtfilt(b,a,xt)

#%%

from pyriemann.estimation import Covariances
covs = Covariances(estimator='oas').fit_transform(X)

from diffusionmap import get_diffusionEmbedding
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

#%%

fig    = plt.figure(figsize=(11, 9.6))
green  = Color("green")
colors = list(green.range_to(Color("red"), int(np.max(lat)+1)))
for i in range(len(u)):
    colori = colors[int(lat[i])].get_rgb()
    plt.scatter(u[i,1], u[i,2], color=colori, s=80)
#plt.xlim(-2,+2)
#plt.ylim(-1,+1)







