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

from utilities.data_analyser import analyse_subject
paramspath = './parameters/physionet.yaml'
pipesdir   = './pipelines/physionet/' 
resultsdir = './results/physionet/' 

nsubjects = 109
for subject in range(1,50+1):
    analyse_subject(subject, paramspath, pipesdir, resultsdir)