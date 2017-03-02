#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:38:24 2017

@author: coelhorp
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('classic')
plt.rc('axes', linewidth=1.5)
plt.rcParams['axes.labelcolor']='k'  
plt.rcParams['xtick.color']='k'   
plt.rcParams['ytick.color']='k'
plt.rcParams['figure.facecolor'] = 'white'

import sys
sys.path.append('../../')

from utilities.data_handler  import get_data
from utilities.dim_reduction import RDR
from pyriemann.utils.distance import distance_riemann, distance_euclid
from pyriemann.estimation import Covariances, CospCovariances
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_riemann

from sklearn.externals import joblib 
from diffusionmap import get_diffusionEmbedding
from scipy.signal import spectrogram
from tqdm import tqdm
from colour import Color

def gen_windows(L, ns, step=1):
    return np.stack([w + np.arange(L) for w in range(0,ns-L+1, step)]) 

for subject in [1, 47, 72, 8]:
    
    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/Physionet/eegmmidb/'
    data_params['session'] = 1
    data_params['task']    = 4
    data_params['fparams'] = [8.0, 35.0] 
    data_params['subject'] = subject
    
    data_params['tparams'] = [-13.8,+13.]           
    X,y = get_data(data_params)
    
    #select = range(2,len(X),3) # avoid taking overlaping epochs
    #X = X[select,:,:]
    #y = y[select]
    
    ##%% Estimate the spectrograms and average them along trials
    #
    #electrodes = [1,7,11,20]
    #spects = []
    #for i in electrodes:
    #    print 'electrode ' + str(i)
    #    S = np.zeros((81, 207))
    #    for t in range(X.shape[0]):
    #        f,t,Sxx = spectrogram(X[t,i,:], fs=160, nperseg=160, noverlap=140)
    #        S = S + Sxx
    #    S = S/X.shape[0]
    #    spects.append(S)
    #
    ##%% Plot the spectrograms as heatmaps
    #
    #fig = plt.figure(figsize=(11.76,9.4))
    #for i,spect in enumerate(spects):
    #    plt.subplot(4,1,i+1)
    #    plt.imshow(spect, aspect='auto', origin='lower')
    #    ax = plt.gca()
    #    plt.ylim(0,20)
    #    #plt.xlim(t[0]-8, t[-1]-16) 
    
    #% Sliding window for covariances and diffusion embedding
    
    L = 160     
    nt,nc,ns = X.shape
    covm = []
    for w in tqdm(gen_windows(L, ns, step=20)):
        xw = X[:,:,w]
        covs = Covariances(estimator='oas').fit_transform(xw)
        covm.append(mean_riemann(covs))
    
    print 'getting the diffusion embedding'
    covm = np.stack(covm)
    u,l = get_diffusionEmbedding(covm, distance_riemann, alpha=1.0, tdiff=0)
    
    filepath  = 'trajectory_real_physionet_subject' + str(subject) + '.pkl'
    embedding = [u,l]
    joblib.dump(embedding, filepath)    

#%% Visualization

filepath  = './trajectory_real_physionet_subject1.pkl'
embedding = joblib.load(filepath) 
u1,l1 = embedding

filepath  = './trajectory_real_physionet_subject8.pkl'
embedding = joblib.load(filepath) 
u8,l8 = embedding

filepath  = './trajectory_real_physionet_subject47.pkl'
embedding = joblib.load(filepath) 
u47,l47 = embedding

filepath  = './trajectory_real_physionet_subject72.pkl'
embedding = joblib.load(filepath) 
u72,l72 = embedding

fig = plt.figure(figsize=(15,15))

ibeg = 65
iend = 135

nplot = 1
for ui,si in zip([u1,u8,u47,u72], [1,8,47,72]):
    ax = fig.add_subplot(2,2,nplot)
    ax.scatter(ui[ibeg,1], ui[ibeg,2], facecolors='none', edgecolors='g', s=200)
    ax.scatter(ui[iend,1], ui[iend,2], facecolors='none', edgecolors='r', s=200)
    green = Color("green")
    colors = list(green.range_to(Color("red"),len(range(ibeg, iend+1))))     
    for i in range(ibeg, iend+1):
        ax.scatter(ui[i,1], ui[i,2], c=colors[i-ibeg].get_rgb(), edgecolors='none', s=120)
    plt.title('Subject ' + str(si), fontsize=22)
    nplot = nplot+1   

#%%










