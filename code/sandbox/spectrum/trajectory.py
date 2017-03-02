#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:03:34 2017

@author: coelhorp
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from numpy.fft    import fft
from numpy.linalg import norm
from scipy.signal import welch, csd

from numpy import cos, pi

from pyriemann.utils.distance import distance_riemann
from pyriemann.estimation import Covariances

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

def gen_sig(ns, SNR=3.0):
   
    falpha = 0.125
    fgamma = 0.250
    fbeta  = 0.375

    r0 = [0.95 for _ in range(600)] + [0.00 for _ in range(1200)] + [0.95 for _ in range(600)]
    r1 = [0.92 for _ in range(2400)]
    r2 = [0.92 for _ in range(2400)]
    r3 = [0.00 for _ in range(1200)] + [0.85 for _ in range(600)] + [0.00 for _ in range(600)]
    r4 = [0.00 for _ in range(1200)] + [0.85 for _ in range(600)] + [0.00 for _ in range(600)]  
    
    x = np.zeros((5,ns))
    for n in range(ns):     
        
        u = np.random.randn(5)                            
        x[0,n] = 2*r0[n]*cos(2*pi*falpha)*x[0,n-1] - r0[n]**2*x[0,n-2] + u[0]           
        x[1,n] = 2*r1[n]*cos(2*pi*fgamma)*x[1,n-1] - r1[n]**2*x[1,n-2] + u[1]
        x[1,n] = x[1,n] + 0.90*x[0,n-1]
        x[2,n] = 2*r2[n]*cos(2*pi*fgamma)*x[2,n-1] - r2[n]**2*x[2,n-2] + u[2]                           
        x[2,n] = x[2,n] + 0.90*x[0,n-1]
        x[3,n] = 2*r3[n]*cos(2*pi*fbeta)*x[3,n-1]  - r3[n]**2*x[3,n-2] + u[3]                           
        x[4,n] = 2*r4[n]*cos(2*pi*fbeta)*x[4,n-1]  - r4[n]**2*x[4,n-2] + u[4]                           
           
    for i in range(3):
        x[i,:] = x[i,:]/np.sqrt(np.var(x[i,:]))
        x[i,:] = x[i,:] + np.sqrt(1.0/SNR)*np.random.randn(ns)
            
    return x

ns = 2400
s = gen_sig(ns, SNR=3.0)

L = 300
def gen_windows(L, ns, step=1):
    return np.stack([w + np.arange(L) for w in range(0,ns-L+1, step)])  

x = []
S = []
for w in gen_windows(L, ns, step=25):
    xw   = s[:,w]
    x.append(xw)
    _,px = welch(xw, nperseg=128)
    S.append(px.T)
S = np.stack(S) 
x = np.stack(x)   

plt.figure(figsize=(12,12))
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.imshow(S[:,:,i].T, aspect='auto')
    plt.grid('off')    

#%%

import sys
sys.path.append('../diffusion/')
from diffusionmap import get_diffusionEmbedding

u_spectrum,_   = get_diffusionEmbedding(x, distance_spectrum)
covs = Covariances().fit_transform(x)
u_covariance,_ = get_diffusionEmbedding(covs, distance_riemann)

#%%

nt = len(u_covariance)
plt.figure(figsize=(13,6))
plt.subplot(1,2,1)
for i in range(nt):
    plt.scatter(u_spectrum[i,1], u_spectrum[i,2], color='b')
plt.subplot(1,2,2)
for i in range(nt):
    plt.scatter(u_covariance[i,1], u_covariance[i,2], color='b')




