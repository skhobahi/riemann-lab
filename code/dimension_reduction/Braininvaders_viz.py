#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:18:14 2017

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

resultsdir = './results/braininvaders/' 

method = []
acc = []
subject = []

nsubjects = 48
for subj in range(1,nsubjects+1):
    rstpaths = resultsdir + 'subject' + str(subj) + '/'
    rstpaths = sorted(glob.glob(rstpaths + '*'))
    for rstpath in rstpaths:
        scores = joblib.load(rstpath)
        method.append(scores['label'])
        acc.append(scores['auc'])
        subject.append(subj)
        
results = pd.DataFrame(data=acc, columns=['AUC'])
results['Method']  = method
results['Subject'] = subject       
methods = method[:4]
       
#%%

import statsmodels.api as sm
from sklearn import linear_model

pairs = [[0,1], [0,2], [0,3]]

methodnames = ['MDM','HRD + MDM','PCA + MDM','bm-RME + MDM']

plt.figure(figsize=(8,8))
plt.subplots_adjust(wspace=0.025, hspace=0.025)
nplot = 1
for pair in pairs:    
        
    pairx,pairy = pair

    method_pair  = [methods[pairx], methods[pairy]]
    results_pair = results[results.Method.isin(method_pair)]
    x = results_pair[results_pair.Method == method_pair[0]].AUC
    y = results_pair[results_pair.Method == method_pair[1]].AUC                    

    if nplot < 3:
        ax = plt.subplot(2,2,nplot)                                  
    else: 
        ax = plt.subplot(2,2,nplot+1) 

    for px,py in zip(x,y):
        plt.scatter(px,py,color='black')
  
    plt.xlim(0,1)
    plt.ylim(0,1)  
    
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], ['', '', '', '', ''])
    if ((nplot-1) % 2) == 0:
        plt.yticks([0, 0.25, 0.50, 0.75, 1.0], ['', '0.25', '', '0.75', ''])

    plt.xticks([0, 0.25, 0.50, 0.75, 1.0], ['', '', '', '', ''])
    if nplot in [1, 3]:
        plt.xticks([0, 0.25, 0.50, 0.75, 1.0], ['', '0.25', '', '0.75', ''])
                            
    lm = linear_model.LinearRegression(fit_intercept=False)
    model_ransac = linear_model.RANSACRegressor(lm)
    model_ransac.fit(x[:len(y),None], y)        
    m = model_ransac.estimator_.coef_[0]   
    plt.text(0.20,0.89,r'$\hat{m} = ' + '{:.4f}'.format(m) + '$', 
             color='black', fontsize=20)  
    
#    model = sm.OLS(y, x[:len(y),None])
#    rst = model.fit()
#    print(rst.f_test("x1 = 1"))     

    plt.plot([0, m],[0, m], color='black', linestyle='--')      
    ax.tick_params(axis='both', which='major', labelsize=16)   
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('both')      
    
    plt.text(0.410, 0.065, methodnames[pairx], fontsize=18, rotation=0) 
    if nplot < 3:
        plt.text(0.055, 0.70, methodnames[pairy], fontsize=18, rotation=90) 
    else:
        plt.text(0.055, 0.77, methodnames[pairy], fontsize=18, rotation=90) 
        
    if nplot in [1, 3]:
        plt.xlabel('AUC', fontsize=18)        
    if nplot in [1, 3]:
        plt.ylabel('AUC', fontsize=18)
    
    nplot = nplot+1

    
plt.savefig('figures/braininvaders-scatterplot.png', format='png')    
plt.savefig('figures/braininvaders-scatterplot.eps', format='eps')    
plt.savefig('figures/braininvaders-scatterplot.pdf', format='pdf')    

































