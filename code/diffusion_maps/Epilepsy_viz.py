#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:54:19 2017

@author: coelhorp
"""

import sys
sys.path.append('../')
import glob
import os

import numpy as np
import scipy as sp
from scipy.signal import welch
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colour import Color
from sklearn.externals import joblib 

def figure_embedding(subject):

    directory = './results/Epilepsy/'
    filepath  = directory + 'embedding_subject-' + subject + '.pkl'
    data = joblib.load(filepath)  
    embedding, lat = data
    u,l = embedding  
    
    # (elev,azim)
    angleparams = {
                    'Dog_3':(-147.84451249999998, -104.21723782991214),
                    'Dog_4':(-172.74685625000001, -72.809613196481052),
                    'Patient_2':(-153.11795000000001, 68.128803225806223),                      
                    'Patient_6':(-141.69216875000001, -49.583806744868411),                            
                    'Patient_7':(-136.7117, -123.4841),
                  }
    
    fig = plt.figure(figsize=(13.73, 9.6), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    red = Color("red")
    latmax = int(np.max(lat)+1)
    colors = list(red.range_to(Color("green"), latmax))
    
    for i,ui in enumerate(u):
        lati = int(lat[i])
        ax.scatter(ui[1], ui[2], ui[3], color=colors[lati].get_rgb(), s=44)
        
    ax.view_init(*angleparams[subject])   
    ax.grid(False)
    ax.set_xlim(-3,+3)
    ax.set_ylim(-3,+3)
    ax.set_zlim(-3,+3)
    ax.set_xticks(range(-2,3))
    ax.set_xticklabels(['-2','-1','0','1','2'])
    ax.set_yticks(range(-2,3))
    ax.set_yticklabels(['-2','-1','0','1','2'])
    ax.set_zticks(range(-2,3))
    ax.set_zticklabels(['-2','-1','0','1','2'])
    
    ttl = plt.title('Embedding for ' + subject, fontsize=26)
    ttl.set_position((0.5, 1.02))
    
    colors_rgb = [color.get_rgb() for color in colors]
    mymap = matplotlib.colors.ListedColormap(colors_rgb, name='from_list')
    sm = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    clb = plt.colorbar(sm)
    clb.set_ticks([0.0, 0.25, 0.50, 0.75, 1.00])
    clb.set_ticklabels([str(lati) for lati in [0, latmax/4, latmax/2, 3*latmax/4, latmax]])
    clb.ax.tick_params(labelsize=18)
    clb.ax.set_ylabel('seconds to seizure', rotation=270, fontsize=20, labelpad=40)
    
    directory = './figures/Epilepsy/'        
    if not os.path.exists(directory):
        os.makedirs(directory)             
    plt.savefig(directory + 'embedding-' + subject + '.eps' , format='eps')        
    plt.savefig(directory + 'embedding-' + subject + '.png' , format='png')        
    plt.savefig(directory + 'embedding-' + subject + '.pdf' , format='pdf')        
    
for subject in ['Dog_3', 'Patient_7', 'Patient_6', 'Dog_4', 'Patient_2']:
    figure_embedding(subject)    






