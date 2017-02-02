#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:53:09 2017

@author: coelhorp
"""

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns

plt.style.use('classic')
plt.rc('axes', linewidth=1.5)
plt.rcParams['axes.labelcolor']='k'  
plt.rcParams['xtick.color']='k'   
plt.rcParams['ytick.color']='k'
plt.rcParams['figure.facecolor'] = 'white'

import sys
sys.path.append('../')

from sklearn.externals import joblib
import glob

paramspath = './parameters/physionet.yaml'
pipesdir   = './pipelines/physionet/' 
resultsdir = './results/physionet/' 

method = []
acc = []

nsubjects = 109
for subject in range(1,nsubjects+1):
    rstpaths = resultsdir + 'subject' + str(subject) + '/'
    rstpaths = glob.glob(rstpaths + '*') 
    for rstpath in rstpaths:
        scores = joblib.load(rstpath)
        method.append(scores['label'])
        acc.append(scores['acc'])
        
results = pd.DataFrame(data=acc, columns=['Accuracy'])
results['Method'] = method

# barplot              
sns.barplot(data=results, y='Method', x='Accuracy', orient='h')
plt.xlim(0.3, 1)
print results.groupby('Method').mean()

#%%

# print along subjects
methods_sup = ['mdm', 'rdr-minmax alpha 1.0 (24) + mdm', 'csp (24) + mdm']
methods_uns = ['mdm', 'rdr-covpca (24) + mdm', 'rdr-hrd-uns (24) + mdm', 'rdr-nrme (24) + mdm']

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
for method in methods_uns:
    scores = results[results.Method == method]['Accuracy']
    scores = sorted(scores, reverse=True)
    plt.plot(scores, label=method)
#plt.legend()    
    
plt.subplot(1,2,2)
for method in methods_sup:
    scores = results[results.Method == method]['Accuracy']
    scores = sorted(scores, reverse=True)
    plt.plot(scores, label=method)
#plt.legend()    


















