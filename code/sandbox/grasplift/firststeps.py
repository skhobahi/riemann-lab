#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:47:47 2017

@author: coelhorp
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info

path = '/localdata/coelhorp/grasplift'
participant = 1
serie = 1

filepath = path + '/P' + str(participant) + '/WS_P' 
filepath = filepath + str(participant) + '_S' + str(serie) + '.mat'
print filepath

mat = sp.io.loadmat(filepath)

i = 0 # which trial to consider
eegi = mat['ws'][0][0]['win']['eeg'][0][i]
emgi = mat['ws'][0][0]['win']['emg'][0][i]

#%%

# define channel type, the first is EEG, the last 6 are stimulations
ch_type  = ['eeg']*32
ch_names = [str(i) for i in range(32)]          

# create and populate MNE info structure
info = create_info(ch_names=ch_names, sfreq=500.0, ch_types=ch_type)
info['filename'] = filepath

# create raw object 
raw = RawArray(eegi.T,info,verbose=False)

#%%

i = 1 # which trial to consider
eegi = mat['ws'][0][0]['win']['eeg'][0][i]
x = eegi[:,12].T
plt.subplot(2,1,i)
plt.plot(x)        

i = 2 # which trial to consider
eegi = mat['ws'][0][0]['win']['eeg'][0][i]
x = eegi[:,12].T
plt.subplot(2,1,i)
plt.plot(x)  

#%%

t,d = raw[12,:]