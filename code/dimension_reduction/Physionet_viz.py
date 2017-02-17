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

method  = []
acc     = []
subject = []
dims    = []
times   = []

nsubjects = 109
for subj in range(1,nsubjects+1):
    rstpaths = resultsdir + 'subject' + str(subj) + '/'
    rstpaths = sorted(glob.glob(rstpaths + '*'))
    for rstpath in rstpaths:
        scores = joblib.load(rstpath)
        method.append(scores['label'])
        acc.append(scores['acc'])
        subject.append(subj)
        times.append(scores['tfit'])
        
        dim = rstpath.split("_")[-2][1:]
        if dim == '':
            dims.append(64)
        else:            
            dims.append(int(rstpath.split("_")[-2][1:]))
        
results = pd.DataFrame(data=acc, columns=['Accuracy'])
results['Method']  = method
results['Subject'] = subject         
results['Dim']     = dims  
results['Times']   = times       
       
#%%    

methodnames = ['hrd-uns', 'minmax','covpca', 'csp', 'hrd-sup']
pvalues = range(4,40,4)

fig = plt.figure()
for meth in methodnames:
    acc = []
    for subject in np.unique(results.Subject):
        rst = results[results.Method.str.contains(meth)]
        rst = rst[rst.Subject == subject]
        acc.append(rst.Accuracy)
    accmean = np.mean(np.stack(acc, axis=0), axis=0)
    plt.plot(pvalues, accmean, linewidth=2.0, label=meth)

accfull = np.mean(results[results.Method == 'mdm'].Accuracy)
plt.plot([4, 36], [accfull, accfull], 
         linestyle='--', color='black', label='full')

plt.ylim(0.55,0.71)    
plt.xlim(pvalues[0], pvalues[-1])
plt.legend(loc="center right", bbox_to_anchor=(1.30,accfull))
plt.xlabel(r"reduced dimension")
plt.ylabel("accuracy")

plt.xticks(pvalues)


#%%

import statsmodels.api as sm
from sklearn import linear_model

p = 24
resultsp = results[(results.Dim == p) | (results.Dim == 64)]

#methodnames = ['mdm', 'hrd-uns', 'minmax', 'covpca', 'hrd-sup']
#pairs = [[0,1], [0,2], [0,3], [0,4]]

methodnames = ['mdm', 'nrme-uns', 'nrme-sup', 'csp']
methodlabel = ['MDM', 'unsRME + MDM', 'supRME + MDM', 'CSP + MDM']
pairs = [[0,1], [0,2], [0,3], [1,2]]


plt.figure(figsize=(8,8))
plt.subplots_adjust(wspace=0.025, hspace=0.025)
nplot = 1
for pair in pairs:    
    pairx,pairy = pair

    method_pair  = [methodnames[pairx], methodnames[pairy]]

    results_x = resultsp[resultsp.Method == 'mdm']
    x = results_x.Accuracy
                    
    results_y = resultsp[resultsp.Method.str.contains(method_pair[1])]
    y = results_y.Accuracy    

    ax = plt.subplot(2,2,nplot)    
    plt.fill_between([0,1],[0,1], color="gray", linewidth=0.0, alpha=0.125)                                                                                             
    for px,py in zip(x,y):
        plt.scatter(px,py,color='black')
  
    plt.xlim(0,1)
    plt.ylim(0,1)  
    
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], ['', '', '', '', ''])
    if ((nplot-1) % 2) == 0:
        plt.yticks([0, 0.25, 0.50, 0.75, 1.0], ['', '0.25', '', '0.75', ''])

    plt.xticks([0, 0.25, 0.50, 0.75, 1.0], ['', '', '', '', ''])
    if (nplot > 2):
        plt.xticks([0, 0.25, 0.50, 0.75, 1.0], ['', '0.25', '', '0.75', ''])
                            
    lm = linear_model.LinearRegression(fit_intercept=False)
    model_ransac = linear_model.RANSACRegressor(lm)
    model_ransac.fit(x[:len(y),None], y)        
    m = model_ransac.estimator_.coef_[0]       
    plt.text(0.27,0.89,r'$\hat{m} = ' + '{:.3f}'.format(m) + '$', 
             color='black', fontsize=20)  
    
    model = sm.OLS(y, x[:len(y),None])
    rst = model.fit()
    print methodnames[pairy] + ' vs ' + methodnames[pairx]
    print(rst.f_test("x1 = 1"))  
    print ''

    plt.plot([0, 1],[0, m], color='black', linestyle='--')      
    ax.tick_params(axis='both', which='major', labelsize=16)   
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('both')      
    
    if nplot in [1, 2]:
        plt.text(0.410, 0.065, methodlabel[pairx], fontsize=18, rotation=0) 
        plt.text(0.055, 0.80, methodlabel[pairy], fontsize=18, rotation=90)     
    if nplot in [3]:
        plt.text(0.410, 0.065, methodlabel[pairx], fontsize=18, rotation=0)         
        plt.text(0.055, 0.70, methodlabel[pairy], fontsize=18, rotation=90)    
    if nplot in [4]:
        plt.text(0.270, 0.065, methodlabel[pairx], fontsize=18, rotation=0)         
        plt.text(0.055, 0.80, methodlabel[pairy], fontsize=18, rotation=90)         
        
    if nplot > 2:
        plt.xlabel('accuracy', fontsize=18)        
    if (nplot-1) % 2 == 0:
        plt.ylabel('accuracy', fontsize=18)
    
    nplot = nplot+1
    
plt.savefig('figures/physionet-scatterplot.png', format='png')    
plt.savefig('figures/physionet-scatterplot.eps', format='eps')       
plt.savefig('figures/physionet-scatterplot.pdf', format='pdf')    

#%%
 
uns = results[results.Method.str.contains('nrme-uns')]
uns = uns.groupby(by='Method').mean()[['Accuracy','Dim']].sort_values(by='Dim')
sup = results[results.Method.str.contains('nrme-sup')]
sup = sup.groupby(by='Method').mean()[['Accuracy','Dim']].sort_values(by='Dim')

plt.plot(uns.Dim, uns.Accuracy, color='b', linewidth=2.0)
plt.plot(sup.Dim, sup.Accuracy, color='r', linewidth=2.0)



















