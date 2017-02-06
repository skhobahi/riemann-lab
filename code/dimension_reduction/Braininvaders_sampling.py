#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:03:57 2017

@author: coelhorp
"""

import glob
import os
import string
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from utilities.data_analyser import (analyse_subject, 
                                     choose_classes,
                                     crossvalidation)

from utilities.data_handler import get_data
from utilities.yaml_handler import (parse_yml_pipeline, 
                                    parse_yml_params,
                                    load_yml)

from sklearn.externals import joblib 
from collections import OrderedDict
from utilities.dim_reduction import RDR
from pyriemann.estimation import Covariances

for subject in range(1,48):

    paramspath = './parameters/braininvaders.yaml'
    pipesdir   = './pipelines/braininvaders/' 
    resultsdir = './results/braininvaders/' 
    
    yml = load_yml(paramspath)
    data_params, analysis_params = parse_yml_params(yml)        
    data_params['subject'] = subject
               
    X,y = get_data(data_params)  
    ntrials = 120
    select = np.random.randint(0, X.shape[0], size=ntrials)
    X = X[select]
    y = y[select]
    
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