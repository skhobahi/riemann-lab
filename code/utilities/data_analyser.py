#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:27:53 2017

@author: coelhorp
"""

import numpy as np

from data_handler import get_data
from helpers.analysis import _crossvalidation
import time

import glob
import os
import string

from yaml_handler import (parse_yml_pipeline, 
                          parse_yml_params,
                          load_yml)

from sklearn.externals import joblib 
from collections import OrderedDict

def choose_classes(labels,classes):    
    
    idx = []
    for cls in classes:        
        idx += np.nonzero(labels == cls)[0].tolist() 
                       
    return idx

def crossvalidation(X, y, pipelines, analysis_params):
    
    nfolds  = analysis_params['nfolds']
    classes = analysis_params['classes']

    idx = choose_classes(y,classes)
    Xc = X[idx]
    yc = y[idx]

    scores = []
    i = 1
    for label,clf in pipelines.items():
        print '   ' + time.asctime(time.localtime(time.time()))
        print '   Pipeline ' + str(i) + ': ' + label 
        score = _crossvalidation(clf, Xc, yc, nfolds)
        score['label'] = label
        scores.append(score)
        print ''
        i = i+1

    return scores     

def analyse_subject(subject, paramspath, pipesdir, resultsdir):

    yml = load_yml(paramspath)
    data_params, analysis_params = parse_yml_params(yml)        
    data_params['subject'] = subject
               
    X,y = get_data(data_params)   
    
    pipelines = OrderedDict()
    pipepaths = sorted(glob.glob(pipesdir + 'pipeline_*'))
    
    for i,ymlpath in enumerate(pipepaths):
        yml = load_yml(ymlpath)
        label = yml['label']
        pipelines[label] = parse_yml_pipeline(yml)   

    print ''
    print 'Processing ' + str(len(pipepaths)),
    print 'pipelines for subject ' + str(subject)
    print ''
    scores = crossvalidation(X, y, pipelines, analysis_params)

    resultsdir = resultsdir + 'subject' + str(subject) + '/'
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)           
                        
    for score,rstpath in zip(scores,pipepaths):
        rstpath = rstpath.split('/')[-1]
        rstpath = string.join([rstpath.split('.')[0]] + ['pkl'], '.')        
        rstpath = resultsdir + rstpath        
        joblib.dump(score, rstpath)     
        
if __name__=='__main__':    
    print 'reloading'
    
    
    
    
    
    
    
    
    
    
    
    
    
    