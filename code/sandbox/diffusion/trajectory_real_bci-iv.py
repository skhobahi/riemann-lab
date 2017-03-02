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

for subject in [1,3,5,7]:
    
    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
    data_params['session'] = 1
    data_params['task']    = 1
    data_params['fparams'] = [8.0, 35.0] 
    data_params['subject'] = subject
    
    data_params['tparams'] = [-8,+16]           
    X,y = get_data(data_params)
    
#    #% Estimate the spectrograms and average them along trials
#    
#    electrodes = [1,7,11,20]
#    spects = []
#    for i in electrodes:
#        print 'electrode ' + str(i)
#        S = np.zeros((126, 180))
#        for t in range(X.shape[0]):
#            f,t,Sxx = spectrogram(X[t,i,:], fs=250, nperseg=250, noverlap=218)
#            S = S + Sxx
#        S = S/X.shape[0]
#        spects.append(S)
#    
#    #% Plot the spectrograms as heatmaps
#    
#    fig = plt.figure(figsize=(11.76,9.4))
#    for i,spect in enumerate(spects):
#        plt.subplot(4,1,i+1)
#        plt.imshow(spect, aspect='auto', origin='lower')
#        plt.ylim(0,20)
#        ax = plt.gca()
        
    #% Sliding window for covariances and diffusion embedding
    
    L = 250     
    nt,nc,ns = X.shape
    covm = []
    for w in tqdm(gen_windows(L, ns, step=32)):
        xw = X[:,:,w]
        covs = Covariances().fit_transform(xw)
        covm.append(mean_riemann(covs))
    
    print 'getting the diffusion embedding'
    covm = np.stack(covm)
    u,l = get_diffusionEmbedding(covm, distance_riemann, alpha=1.0, tdiff=0)
    
    filepath  = 'trajectory_real_bci-iv_subject' + str(subject) + '.pkl'
    embedding = [u,l]
    joblib.dump(embedding, filepath)    

#%% Visualization

   











