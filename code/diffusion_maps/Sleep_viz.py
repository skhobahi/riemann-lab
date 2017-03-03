#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:36:27 2017

@author: coelhorp
"""

import os
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
    
    fig = plt.figure(figsize=(11.875, 10.175), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    
    ax.view_init(23.84765625, -70.293255131964827) 
    colors = ['r','g','b','b'] # merge S1 and S2 into the same class
    for i in range(len(u)):
        ax.scatter(u[i,1], u[i,2], u[i,3], color=colors[y[i]], s=44)     
    ax.grid(False)
    ax.set_xlim(-3,+3)
    ax.set_ylim(-3,+3)
    ax.set_zlim(-4,+4)
    ax.set_xticks(range(-2,3))
    ax.set_xticklabels(['-2','-1','0','1','2'])
    ax.set_yticks(range(-2,3))
    ax.set_yticklabels(['-2','-1','0','1','2'])
    ax.set_zticks(range(-3,4))
    ax.set_zticklabels(['-3','-2','-1','0','1','2','3'])
    
    hnd = []
    for colori in colors:
        hnd.append(plt.scatter([],[],color=colori, s=44))
    plt.legend(hnd, ('S3', 'RM','S1','S2'), loc=(0.92, 0.075))  
        
    directory = './figures/Sleep/'        
    if not os.path.exists(directory):
        os.makedirs(directory)             
    plt.savefig(directory + 'embedding.eps' , format='eps')        
    plt.savefig(directory + 'embedding.png' , format='png')        
    plt.savefig(directory + 'embedding.pdf' , format='pdf')          
        
figure_embedding()   

  