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

from diffusionmap import get_diffusionEmbedding


data_params = {}
data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
data_params['session'] = 1
data_params['task']    = 1
data_params['fparams'] = [8.0, 35.0] 
data_params['subject'] = 1

data_params['tparams'] = [-3.0, -1.0]           
X,y = get_data(data_params)
covs1 = Covariances().fit_transform(X)     
covs1 = covs1[(y == 1) | (y == 2)]      
           
data_params['tparams'] = [1.5, 3.75]         
X,y = get_data(data_params)
covs2 = Covariances().fit_transform(X)
covs2 = covs2[(y == 1) | (y == 2)]      

data_params['tparams'] = [6.0, 8.0]         
X,y = get_data(data_params)
covs3 = Covariances().fit_transform(X)
covs3 = covs3[(y == 1) | (y == 2)]      

y = y[(y == 1) | (y == 2)]
covsleft  = covs2[y == 1]      
covsright = covs2[y == 2]      

#%%

colors = ['g' for _ in range(len(covs1))]
colors = colors + ['r' for _ in range(len(covsleft))]
colors = colors + ['b' for _ in range(len(covsright))]

covs = np.concatenate((covs1, covsleft, covsright), axis=0)
covs = np.delete(covs, (22), axis=0) 
covs = np.delete(covs, (48), axis=0) 
u,l = get_diffusionEmbedding(covs, distance_riemann, alpha=1.0, tdiff=0)

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,ui in enumerate(u):
    ax.scatter(ui[1], ui[2], ui[3], color=colors[i])
#    plt.text(ui[1], ui[2], str(i))



