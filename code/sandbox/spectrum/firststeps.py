#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:47:51 2017

@author: coelhorp
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from numpy.fft    import fft
from numpy.linalg import norm
from scipy.signal import welch, csd

from numpy import cos, pi

from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
    
def gen_sig(ns, SNR=3.0, condition=1):

    params = [(0.125, 0.250, 0.375, 0.5, 0.5), 
              (0.125, 0.250, 0.375, 0.0, 0.0), 
              (0.125, 0.250, 0.000, 0.5, 0.5),
              (0.125, 0.000, 0.375, 0.0, 0.0)]
    
    f0,f1,f2,a20,a21 = params[condition-1]
    r = 0.95

    x = np.zeros((3,ns))
    for n in range(ns):       
        u = np.random.randn(3)                
            
        x[0,n] = 2*r*cos(2*pi*f0)*x[0,n-1] - r**2*x[0,n-2] + u[0]           
        x[1,n] = 2*r*cos(2*pi*f1)*x[1,n-1] - r**2*x[1,n-2] + u[1]   
        x[2,n] = 2*r*cos(2*pi*f2)*x[2,n-1] - r**2*x[2,n-2] + u[2]                           
        x[2,n] = x[2,n] + a20*x[0,n-1] + a21*x[1,n-1]              
           
    for i in range(3):
        x[i,:] = x[i,:]/np.sqrt(np.var(x[i,:]))
        x[i,:] = x[i,:] + np.sqrt(1.0/SNR)*np.random.randn(ns)
            
    return x
        
def cross_spectrum(x, NFFT=64):
    
    nc,ns = x.shape
    S = np.zeros((NFFT/2+1, nc, nc), dtype=complex)
    for i in range(nc):
        for j in range(nc):
            sx = x[i,:]
            sy = x[j,:]
            fss,pss = csd(sx,sy,nperseg=NFFT)   
            S[:,i,j] = pss
        
    return fss, S       

def distance_spectrum(x, y):
    
    _,px = cross_spectrum(x)
    _,py = cross_spectrum(y)    
    
    df = 0
    for f in range(len(px)):
        df = df + distance_riemann(px[f], py[f])**2
    
    return df
    
ns = 300
nt = 100

x = []
y = []
for _ in range(nt/2):
    x.append(gen_sig(ns, condition=2))
    y.append(1)
for _ in range(nt/2):
    x.append(gen_sig(ns, condition=4))
    y.append(2)
x = np.stack(x, axis=0)

#%%

import sys
sys.path.append('../diffusion/')
from diffusionmap import get_diffusionEmbedding

u_spectrum,_   = get_diffusionEmbedding(x, distance_spectrum)
covs = Covariances().fit_transform(x)
u_covariance,_ = get_diffusionEmbedding(covs, distance_riemann)

#%%

plt.figure(figsize=(13,6))
plt.subplot(1,2,1)
for i in range(nt):
    plt.scatter(u_spectrum[i,1], u_spectrum[i,2], color=['b','r'][y[i]-1])
plt.subplot(1,2,2)
for i in range(nt):
    plt.scatter(u_covariance[i,1], u_covariance[i,2], color=['b','r'][y[i]-1])









