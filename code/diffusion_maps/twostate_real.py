#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:08:18 2017

@author: coelhorp
"""

import sys
sys.path.append('../../')
from utilities.diffusion_map import get_diffusionEmbedding

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('classic')
plt.rc('axes', linewidth=1.5)
plt.rcParams['axes.labelcolor']='k'  
plt.rcParams['xtick.color']='k'   
plt.rcParams['ytick.color']='k'
plt.rcParams['figure.facecolor'] = 'white'

from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../../')

from utilities.data_handler  import get_data
from utilities.dim_reduction import RDR
from pyriemann.utils.distance import distance_riemann
from pyriemann.estimation import Covariances, ERPCovariances

def savefigure(name):
    plt.savefig(name + '.png', format='png')    
    plt.savefig(name + '.eps', format='eps')       
    plt.savefig(name + '.pdf', format='pdf')      

def make_fig1(save=False):

    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/Physionet/eegmmidb/'
    data_params['session'] = 1
    data_params['task']    = 4
    data_params['tparams'] = [1.0, 2.0]
    data_params['fparams'] = [8.0, 35.0]  
    
    data_params['subject'] = 47
    X,yworst = get_data(data_params)
    covs = Covariances().fit_transform(X)
    uworst,lworst = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    data_params['subject'] = 72
    X,ybest = get_data(data_params)
    covs = Covariances().fit_transform(X)
    ubest,lbest = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    fig = plt.figure(figsize=(10.5,5))
    plt.subplots_adjust(wspace=0.020, hspace=0.025)
    
    plt.subplot(1,2,1)
    colorst = [['b', 'r'][int(t)] for t in (yworst-2)]  
    plt.scatter(uworst[:,1], uworst[:,2], color=colorst, s=44)
    plt.xlabel(r'$\phi_1$', fontsize=26)
    plt.ylabel(r'$\phi_2$', fontsize=26)
    plt.xticks([])
    plt.yticks([])
    ttl = plt.title('Worst subject', fontsize=20)
    ttl.set_position([.5, 1.025])
    
    ax = plt.subplot(1,2,2)
    colorst = [['b', 'r'][int(t)] for t in (ybest-2)]  
    plt.scatter(ubest[:,1], ubest[:,2], color=colorst, s=44)
    plt.xlabel(r'$\phi_1$', fontsize=26)
    plt.ylabel(r'$\phi_2$', fontsize=26)
    ax.yaxis.set_label_position("right")
    plt.xticks([])
    plt.yticks([])
    ttl = plt.title('Best subject', fontsize=20)
    ttl.set_position([.5, 1.025])
    
    if save:    
        name = 'figure1'
        savefigure(name)        
    
    return [uworst, lworst], [ubest, lbest]

def make_fig2(save=False):

    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
    data_params['session'] = 1
    data_params['task']    = 1
    data_params['tparams'] = [1.25, 3.75]
    data_params['fparams'] = [8.0, 35.0]  
    
    data_params['subject'] = 5
    X,yworst = get_data(data_params)
    X = X[(yworst == 1) | (yworst == 2)]
    yworst = yworst[(yworst == 1) | (yworst == 2)]
    covs = Covariances().fit_transform(X)
    uworst,lworst = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    data_params['subject'] = 1
    X,ybest = get_data(data_params)
    X = X[(ybest == 1) | (ybest == 2)]
    ybest = ybest[(ybest == 1) | (ybest == 2)]
    covs = Covariances().fit_transform(X)
    ubest,lbest = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    fig = plt.figure(figsize=(10.5,5))
    plt.subplots_adjust(wspace=0.020, hspace=0.025)
    
    plt.subplot(1,2,1)
    colorst = [['b', 'r'][int(t)] for t in (yworst-2)]  
    plt.scatter(uworst[:,1], uworst[:,2], color=colorst, s=44)
    plt.xlabel(r'$\phi_1$', fontsize=26)
    plt.ylabel(r'$\phi_2$', fontsize=26)
    plt.xticks([])
    plt.yticks([])
    ttl = plt.title('Worst subject', fontsize=20)
    ttl.set_position([.5, 1.025])
    
    ax = plt.subplot(1,2,2)
    colorst = [['b', 'r'][int(t)] for t in (ybest-2)]  
    plt.scatter(ubest[:,1], ubest[:,2], color=colorst, s=44)
    plt.xlabel(r'$\phi_1$', fontsize=26)
    plt.ylabel(r'$\phi_2$', fontsize=26)
    ax.yaxis.set_label_position("right")
    plt.xticks([])
    plt.yticks([])
    ttl = plt.title('Best subject', fontsize=20)
    ttl.set_position([.5, 1.025])
    
    if save:    
        name = 'figure2'
        savefigure(name)     

    return [uworst, lworst], [ubest, lbest]

def make_fig3(save=False):

    data_params = {}
    data_params['path'] = '/research/vibs/Pedro/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
    data_params['task']    = 1
    data_params['tparams'] = [1.25, 3.75]
    data_params['fparams'] = [8.0, 35.0]  
    data_params['subject'] = 1
               
    data_params['session'] = 1           
    X1,y1 = get_data(data_params)
    X1 = X1[(y1 == 1) | (y1 == 2)]
    y1 = y1[(y1 == 1) | (y1 == 2)]
    
    data_params['session'] = 2           
    X2,y2 = get_data(data_params)
    X2 = X2[(y2 == 1) | (y2 == 2)]
    y2 = y2[(y2 == 1) | (y2 == 2)]
    X2 = np.delete(X2, (37), axis=0) # delete bad trial
    y2 = np.delete(y2, (37), axis=0) # delete bad trial
    
    y = np.concatenate((y1, y2), axis=0)
    
    covs  = Covariances().fit_transform(X1)
    u1,l1 = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    covs  = Covariances().fit_transform(X2)
    u2,l2 = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    covs = Covariances().fit_transform(np.concatenate((X1,X2), axis=0))
    u,l = get_diffusionEmbedding(points=covs, distance=distance_riemann)
    
    fig1 = plt.figure(figsize=(15.5,5))
    plt.subplots_adjust(wspace=0.020, hspace=0.025)
    
    ax1 = plt.subplot(1,3,1)
    colorst  = [['b', 'r'][int(t)] for t in (y1-2)]  
    for i,ui in enumerate(u1):
        plt.scatter(ui[1], ui[2], color=colorst[i], s=44)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.xlabel(r'$\phi_1$', fontsize=26)
    plt.ylabel(r'$\phi_2$', fontsize=26)
    ttl = plt.title('Session 1', fontsize=24)
    ttl.set_position([.5, 1.025])
        
    ax2 = plt.subplot(1,3,2)
    colorst  = [['b', 'r'][int(t)] for t in (y2-2)]  
    for i,ui in enumerate(u2):
        plt.scatter(ui[1], ui[2], color=colorst[i], s=44)  
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.xlabel(r'$\phi_1$', fontsize=26)
    ttl = plt.title('Session 2', fontsize=24)
    ttl.set_position([.5, 1.025])
    
    ax3 = plt.subplot(1,3,3)
    colorst  = [['b', 'r'][int(t)] for t in (y-2)]  
    markerst = ['o' for _ in range(len(y1))] + ['*' for _ in range(len(y2))]
    for i,ui in enumerate(u):
        plt.scatter(ui[1], ui[2], color=colorst[i], s=44, marker=markerst[i])  
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.xlabel(r'$\phi_1$', fontsize=26)
    plt.ylabel(r'$\phi_2$', fontsize=26)
    ax3.yaxis.set_label_position("right")
    ttl = plt.title('Both sessions', fontsize=24)
    ttl.set_position([.5, 1.025])
    
    if save:    
        name = 'figure3a'
        savefigure(name)     
    
    fig2 = plt.figure(figsize=(9,8))
    ax = fig2.add_subplot(111, projection='3d')
    
    colorst  = [['b', 'r'][int(t)] for t in (y-2)]  
    markerst = ['o' for _ in range(len(y1))] + ['*' for _ in range(len(y2))]
    for i,ui in enumerate(u):
        ax.scatter(ui[1], ui[2], ui[3], color=colorst[i], s=44, marker=markerst[i])
    ax.set_xlabel(r'$\phi_1$', fontsize=26)
    ax.set_ylabel(r'$\phi_2$', fontsize=26)
    ax.set_zlabel(r'$\phi_3$', fontsize=26)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ttl = plt.title('Both sessions', fontsize=24)
    ttl.set_position([0.50, 0.975])
    
    if save:    
        name = 'figure3b'
        savefigure(name)      

#_ = make_fig1(save=True)
_ = make_fig2(save=False)
#_ = make_fig3(save=True)


