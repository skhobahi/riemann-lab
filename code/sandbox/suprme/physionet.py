#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:52:49 2017

@author: coelhorp
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from utilities.data_handler  import get_data
from utilities.dim_reduction import RDR
from pyriemann.utils.distance import distance_riemann
from pyriemann.estimation import Covariances

from pyriemann.utils.base     import powm, sqrtm, invsqrtm, logm

def dim_reduction_nrme_sup(X, P, labels):
    
    K  = X.shape[0]
    nc = X.shape[1]    
    
    Sw = np.zeros((nc,nc))
    Sb = np.zeros((nc,nc))
    for i in range(K):
        ci = labels[i]
        for j in range(K):
            Ci, Cj = X[i,:,:], X[j,:,:]         
            Sij = np.dot(invsqrtm(Ci), np.dot(Cj, invsqrtm(Ci)))          
            if (i != j) & (labels[j] == ci):            
                Sw  = Sw + powm(logm(Sij), 2)
            if (i != j) & (labels[j] != ci):             
                Sb  = Sb + powm(logm(Sij), 2)            
    
    M = np.dot(np.linalg.inv(Sw),Sb)
    g,U = np.linalg.eig(M)        
    
    idx = g.argsort()[::-1]
    g = g[idx]
    U = U[:,idx]
    
    B,p = sp.linalg.polar(U)
    W = B[:,:P]   
    
    return W

data_params = {}
data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/Physionet/eegmmidb/'
data_params['session'] = 1
data_params['task']    = 4
data_params['tparams'] = [1.0, 2.0]
data_params['fparams'] = [8.0, 35.0] 
data_params['subject'] = 1
           
X,y  = get_data(data_params)
covs = Covariances().fit_transform(X)           

P = 24
W = dim_reduction_nrme_sup(covs, P, y)
 

   
















