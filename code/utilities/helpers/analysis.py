#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:59:46 2016

@author: coelhorp
"""

import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
import time

def _crossvalidation(pipeline, X, y, nfolds):      
    
    labels = np.unique(y)
    cv = StratifiedKFold(n_splits=nfolds)
    splits = cv.split(X, y)    

    acc = []
    auc = []   
    cfm = []        
    
    tfit  = []
    tpred = []
           
    print '   [',              
    for split in splits:                
        
        print '.',
        train, valid = split        
        Xtrain, ytrain = X[train, :, :], y[train]
        Xvalid, yvalid = X[valid, :, :], y[valid]        
        
        t = time.time()
        pipeline.fit(Xtrain, ytrain)                 
        tfit.append(time.time() - t)
        
        t = time.time()
        ypred = pipeline.predict(Xvalid)
        tpred.append(time.time() - t)

        acc.append(pipeline.score(Xvalid, yvalid))   
        
        yvalid_ = _convertlabels(yvalid, labels)
        ypred_  = _convertlabels(ypred, labels)
        auc.append(roc_auc_score(yvalid_, ypred_))                        
        
        cfm.append(confusion_matrix(yvalid, ypred))
            
    print ']'                         

    acc_avg = np.mean(acc)
    auc_avg = np.mean(auc)
    cfm_avg = sum(cfm)/float(len(cfm)) 
       
    tfit_avg  = np.mean(tfit)
    tpred_avg = np.mean(tpred)
    
    scores = {}               
    scores['acc'] = acc_avg
    scores['auc'] = auc_avg
    scores['cfm'] = cfm_avg
    scores['tfit']  = tfit_avg
    scores['tpred'] = tpred_avg
                                    
    return scores  

def _convertlabels(y, labels):     
    y_true = np.zeros((y.shape[0], len(labels)))   
      
    for i, label in enumerate(labels):
        y_true[:, i] = (y == label)
                
    return y_true             
    










    
    
