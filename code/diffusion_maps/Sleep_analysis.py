#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:36:27 2017

@author: coelhorp
"""

import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann
from mpl_toolkits.mplot3d import Axes3D

from utilities.diffusion_map import get_diffusionEmbedding
from sklearn.externals import joblib 

def get_embedding():

    filepath = '/localdata/coelhorp/sleepdata/cov_mtrs_ins3_2sec.npy'
    (cov_mtrs,stages) = np.load(filepath)
    
    ntrials = 200
    X = []
    y = []
    for i,stage in enumerate(['S3', 'RM','S1','S2']):
        Xi = cov_mtrs[stages[stage]][:ntrials]
        X.append(Xi)
        y = y + [i for _ in range(len(Xi))]
    covs = np.concatenate(X, axis=0)        
    
    print 'getting the diffusion embedding'    
    u,l = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    filepath  = './results/Sleep/'
    filepath  = filepath + 'embedding.pkl'
    embedding = [u,l]
    joblib.dump([embedding, y], filepath)      

#get_embedding()



      
