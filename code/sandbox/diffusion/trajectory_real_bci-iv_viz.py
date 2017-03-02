#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:50:43 2017

@author: coelhorp
"""

import matplotlib.pyplot as plt
from colour import Color
from sklearn.externals import joblib 

filepath  = './trajectory_real_bci-iv_subject1.pkl'
embedding = joblib.load(filepath) 
u1,l1 = embedding

filepath  = './trajectory_real_bci-iv_subject3.pkl'
embedding = joblib.load(filepath) 
u3,l3 = embedding

filepath  = './trajectory_real_bci-iv_subject5.pkl'
embedding = joblib.load(filepath) 
u5,l5 = embedding

filepath  = './trajectory_real_bci-iv_subject7.pkl'
embedding = joblib.load(filepath) 
u7,l7 = embedding


#%% Compare the trajectories for 4 subjects



fig = plt.figure(figsize=(15,15))

nplot = 1
for ui,si in zip([u1, u3, u5, u7], [1, 3, 5, 7]):
    ax = fig.add_subplot(2,2,nplot)
    ax.scatter(ui[i_000,1], ui[i_000,2], facecolors='none', edgecolors='g', s=200)
    ax.scatter(ui[i_p08,1], ui[i_p08,2], facecolors='none', edgecolors='r', s=200)
    green = Color("green")
    colors = list(green.range_to(Color("red"),len(range(i_000, i_p08+1))))     
    for i in range(i_000, i_p08+1):
        ax.scatter(ui[i,1], ui[i,2], c=colors[i-i_000].get_rgb(), edgecolors='none', s=120)
    plt.title('Subject ' + str(si), fontsize=22)
    nplot = nplot+1


    