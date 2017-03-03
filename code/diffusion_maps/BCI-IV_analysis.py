#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:38:24 2017

@author: coelhorp
"""

import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('classic')
plt.rc('axes', linewidth=1.5)
plt.rcParams['axes.labelcolor']='k'  
plt.rcParams['xtick.color']='k'   
plt.rcParams['ytick.color']='k'
plt.rcParams['figure.facecolor'] = 'white'

from utilities.data_handler  import get_data
from utilities.dim_reduction import RDR
from utilities.diffusion_map import get_diffusionEmbedding

from pyriemann.utils.distance import distance_riemann, distance_euclid
from pyriemann.estimation import Covariances, CospCovariances
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_riemann

from sklearn.externals import joblib 
from scipy.signal import spectrogram
from tqdm import tqdm
from colour import Color

def gen_windows(L, ns, step=1):
    return np.stack([w + np.arange(L) for w in range(0,ns-L+1, step)])

# use a sliding window of 1s to see how the covariance matrices evolve in time
# at each timeframe I take the geometric mean of the covariance matrices along the trials
def get_trajectory(subject):

    print 'subject ' + str(subject)
    print ''
    
    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
    data_params['session'] = 1
    data_params['task']    = 1
    data_params['fparams'] = [8.0, 35.0] 
    data_params['subject'] = subject    
    data_params['tparams'] = [-8,+16]           

    X,y = get_data(data_params)
        
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
    
    filepath  = './results/BCI-IV/'
    filepath  = filepath + 'trajectory_subject' + str(subject) + '.pkl'
    embedding = [u,l]
    joblib.dump(embedding, filepath)    
    
    print ''

# embed the trials from two classes and see how well separated they are   
def get_twoclasses(subject):
    
    print 'subject ' + str(subject)
    print ''    

    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
    data_params['session'] = 1
    data_params['task']    = 1
    data_params['tparams'] = [1.25, 3.75]
    data_params['fparams'] = [8.0, 35.0] 
    data_params['subject'] = subject
               
    X,y = get_data(data_params)
    X = X[(y == 1) | (y == 2)]
    y = y[(y == 1) | (y == 2)]
    covs = Covariances().fit_transform(X)
    u,l = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    filepath  = './results/BCI-IV/'
    filepath  = filepath + 'twoclasses_subject' + str(subject) + '.pkl'
    embedding = [u,l]
    joblib.dump([embedding, y], filepath) 
    
#[get_trajectory(subject) for subject in [1,3,5,7]]    
#[get_twoclasses(subject) for subject in [1,3,5,7]]    








