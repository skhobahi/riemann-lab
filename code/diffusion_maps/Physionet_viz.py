#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:50:43 2017

@author: coelhorp
"""

import matplotlib.pyplot as plt
from colour import Color
from sklearn.externals import joblib 
import glob

def figure_trajectory():

    path  = './results/Physionet/'
    filepaths = glob.glob(path + '*trajectory*') 
    u = []
    l = []
    subjects = []
    for filepath in filepaths:    
        embedding = joblib.load(filepath) 
        s = filepath.split('.')[-2].split('subject')[-1]
        subjects.append(int(s))
        u.append(embedding[0])
        l.append(embedding[1])
    
    plt.figure(figsize=(15.1,14.61))
    i_000 = 65  # beginning of the window of interest (at cue) 
    i_p08 = 135 # end of the window of interest (8s after cue)
    green = Color("green")
    colors = list(green.range_to(Color("red"),len(range(i_000, i_p08+1))))  
        
    nplot = 1
    for subj,ui in zip(subjects,u):    
        plt.subplot(2,2,nplot)      
        for i in range(i_000, i_p08+1):
            plt.scatter(ui[i,1], ui[i,2], color=colors[i-i_000].get_rgb(), s=120)
        nplot = nplot + 1
        plt.title('Subject ' + str(subj), fontsize=22)
        plt.scatter(ui[i_000,1], ui[i_000,2], color='g', s=250, marker='s')
        plt.scatter(ui[i_p08,1], ui[i_p08,2], color='r', s=250, marker='s')     

def figure_twoclasses():

    path  = './results/Physionet/'
    filepaths = glob.glob(path + '*twoclasses*') 
    u = []
    l = []
    y = []
    subjects = []
    for filepath in filepaths:    
        data = joblib.load(filepath) 
        embedding, labels = data    
        s = filepath.split('.')[-2].split('subject')[-1]
        subjects.append(int(s))
        u.append(embedding[0])
        l.append(embedding[1])
        y.append(labels)
    
    plt.figure(figsize=(15.1,14.61))
    colors = ['r', 'b']    
    nplot = 1
    for subj,ui,yi in zip(subjects,u,y):    
        plt.subplot(2,2,nplot)      
        for i in range(len(ui)):
            plt.scatter(ui[i,1], ui[i,2], color=colors[yi[i]-2], s=80)
        nplot = nplot + 1
        plt.title('Subject ' + str(subj), fontsize=22)









