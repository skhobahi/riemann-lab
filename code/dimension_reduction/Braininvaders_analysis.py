#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:53:09 2017

@author: coelhorp
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')

from utilities.data_analyser import analyse_subject, choose_classes  

from utilities.data_handler import get_data
from utilities.yaml_handler import (parse_yml_pipeline, 
                                    parse_yml_params,
                                    load_yml)

from utilities.dim_reduction import RDR

from pyriemann.estimation import Covariances

paramspath = './parameters/braininvaders.yaml'
pipesdir   = './pipelines/braininvaders/' 
resultsdir = './results/braininvaders/' 

nsubjects = 48
for subject in range(1,nsubjects+1):
    analyse_subject(subject, paramspath, pipesdir, resultsdir)














