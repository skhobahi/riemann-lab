#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:36:27 2017

@author: coelhorp
"""

import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib 

def figure_embedding():

    path  = './results/Sleep/'
    filepaths = glob.glob(path + '*embedding*') 
    
    data = joblib.load(filepaths[0]) 
    embedding,y = data
    u = embedding[0]
    l = embedding[1]
    
    fig = plt.figure(figsize=(11, 9.6))
    ax  = fig.add_subplot(111, projection='3d')
    colors = ['r','g','b','b'] # merge S1 and S2 into the same class
    for i in range(len(u)):
        ax.scatter(u[i,1], u[i,2], u[i,3], color=colors[y[i]], s=44)