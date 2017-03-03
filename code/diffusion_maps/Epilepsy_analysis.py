#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:54:19 2017

@author: coelhorp
"""

import sys
sys.path.append('../')
import glob
import os

import numpy as np
import scipy as sp
from scipy.signal import welch
from scipy import signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colour import Color
from sklearn.externals import joblib 

from pyriemann.utils.distance import distance_riemann
from utilities.diffusion_map import get_diffusionEmbedding
from pyriemann.estimation import Covariances

def get_embedding():

    path  = '/localdata/coelhorp/epilepsy/seizure_detection/'
    
    subjects = ['Dog_3', 'Dog_4', 'Patient_2', 'Patient_6', 'Patient_7']
    for subject in subjects:
        
        print 'processing subject: ' + subject    
        filepaths = sorted(glob.glob(path + subject + '/*_ictal_*'))
    
        X = []
        lat = []
        for filepath in filepaths:
            struct = sp.io.loadmat(filepath)
            X.append(struct['data'])  
            lat.append(struct['latency'][0])
            
        lat = np.array(lat)    
        X = np.stack(X)    
    
        fs = struct['freq']
        fini = 1.0
        fend = 40.0
        b,a = signal.butter(5, [fini/(fs/2), fend/(fs/2)], btype='bandpass')
        for xt in X:
            f,pxx = welch(xt, fs=fs)
            xt = signal.filtfilt(b,a,xt)
    
        covs = Covariances(estimator='oas').fit_transform(X)
        print 'getting the diffusion embedding'
        u,l = get_diffusionEmbedding(points=covs, distance=distance_riemann)
        
        directory = './results/Epilepsy/'        
        if not os.path.exists(directory):
            os.makedirs(directory)        
    
        filepath  = directory + 'embedding_subject-' + str(subject) + '.pkl'
        embedding = [u,l]
        joblib.dump([embedding, lat], filepath)    
        
        print ''            

#get_embedding()







