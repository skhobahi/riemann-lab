#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:36:43 2016

@author: coelhorp
"""

import yaml
import importlib

from sklearn.pipeline import make_pipeline

def load_yml(filepath):
	"""Loads a YAML file"""
	with open(filepath,"r") as file_descriptor:
		data = yaml.safe_load(file_descriptor)
	return data

def parse_yml_params(yml):   
    
    path = yml['dataparams']['path']
    
    # some datasets have two sessions or more
    session = yml['dataparams']['session']  
    
    # specify the task in which we are interested
    try:
        # some datasets have more than one task
        task = yml['dataparams']['task'] 
    except KeyError:
        # user may not want to specify a task 
        task = 1               
        
    # limits that define an epoch (referenced to cue)
    tparams = yml['dataparams']['tparams'] 
    # low and high cutoff frequencies for filtering - 4th order Butterworth
    fparams = yml['dataparams']['fparams'] 
    # number of folds for cross-valid
    nfolds = yml['analysisparams']['crossvalidation']['nfolds']    
    # which classes to consider
    classes = yml['analysisparams']['classes']       
    
    data_params = {}
    data_params['path']    = path
    data_params['session'] = session
    data_params['task']    = task
    data_params['tparams'] = tparams
    data_params['fparams'] = fparams

    analysis_params = {}          
    analysis_params['nfolds']  = nfolds
    analysis_params['classes'] = classes
      
    return data_params, analysis_params  
    
def parse_yml_pipeline(yml):
    
    # creates a new auxiliary dictionnary from the loaded yml
    est_dict = {}
    for pkg_str, functions in yml['imports'].iteritems():
        for func_str in functions:
            pkg  = importlib.import_module(pkg_str)
            func = getattr(pkg,func_str) 
            est_dict[func_str] = func

    # creates a pipeline using the auxiliary dic from above
    pipe = []
    for step_dict in yml['pipeline']:
        step_str = step_dict.keys()[0]

        if step_dict[step_str] is not None:        
            step_est = est_dict[step_str](**step_dict[step_str])
        else:
            step_est = est_dict[step_str]()
            
        pipe.append(step_est)
        
    return make_pipeline(*pipe)   

if __name__=='__main__':    
    print 'reloading'