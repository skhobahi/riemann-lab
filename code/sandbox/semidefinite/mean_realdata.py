#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:00:11 2017

@author: coelhorp
"""

import numpy as np
import sys
sys.path.append('../../')

from utilities.getdata     import data_import
from utilities.preparedata import data_prepare, choose_classes
from pyriemann.estimation  import Covariances
from pyriemann.utils.mean  import mean_riemann, mean_euclid
from pyriemann.utils.distance import distance_riemann

from numpy.linalg import eig

import matplotlib.pyplot as plt

def lowrank_approx(A, p):
    u,s,v = np.linalg.svd(A)
    u_ = u[:,:p]
    s_ = s[:p]
    v_ = v.T
    v_ = v_[:,:p]
    return np.dot(u_, np.dot(np.diag(s_), v_.T))

def eigendecomp_rank(X, p):

    w,v = np.linalg.eig(X)    
    idx = w.argsort()[::-1]

    w = np.real(w[idx])
    v = np.real(v[:,idx])
    
    Rsq = np.diag(w[:p])
    U  = v[:,:p]    
    
    return Rsq, U  
    
def mean_subspace(U):  
    
    n,p = U[0].shape
    sumU = np.zeros((n,n))
    for Ui in U:
        sumU += np.dot(Ui, Ui.T)
    u,s,v = np.linalg.svd(sumU) 
    v = v.T
    return v[:,:p]
     
def mean_semidefinite(A, p):    

    Rsq = []
    U = []          
    for Ai in A:    
        Rsqi, Ui = eigendecomp_rank(Ai, p)
        Rsq.append(Rsqi)
        U.append(Ui)
    W = mean_subspace(U)    
    
    T = []
    for Ai, Ui in zip(A, U):
        Oi,_,OiWt = np.linalg.svd(np.dot(Ui.T, W))
        OiW = OiWt.T
        Yi = np.dot(Ui, Oi)
        Wi = np.dot(W, OiW)    
        Si2 = np.dot(Yi.T, np.dot(Ai, Yi))
        _ = np.dot(Wi, np.dot(Si2, Wi.T))
        Ti2 = np.dot(W.T, np.dot(_, W))
        T.append(Ti2)
    M = mean_riemann(np.stack(T))
    avg = np.dot(W, np.dot(M, W.T)) 
    
    return avg     

#path = '/localdata/coelhorp/datasets/motorimagery/PhysioNet/eegmmidb/'
#subject  = 7
#sessions = [1]
#tparams  = [1., 2.]
#fparams  = [7.0,35.0] 
#classes  = [2,3]
#task = 4

#path = '/localdata/coelhorp/datasets/motorimagery/BCI-competitions/BCI-III/IVa/'
#subject  = 1
#sessions = [1]
#tparams  = [0.5, 2.5]
#fparams  = [8.0,35.0] 
#classes  = [1,2]
#task = 1

path = '/localdata/coelhorp/datasets/motorimagery/BCI-competitions/BCI-IV/2a/'
subject  = 1
sessions = [1]
tparams  = [1.25, 3.75]
fparams  = [8.0,35.0] 
classes  = [1, 2, 3, 4]
task = 1

raw, event_id = data_import(path, subject, sessions, task)

epochs, labels = data_prepare(raw, tparams, fparams, 
                              events_interest=event_id)

# choose subset of data according to classes of interest
idx = choose_classes(epochs, labels, classes)
X = epochs[idx,:,:]
y = labels[idx]

cov = Covariances().fit_transform(X)

#%%

class1 = classes[0]
class2 = classes[1]

C_1 = cov[labels == class1]
C_2 = cov[labels == class2]
M_1 = mean_riemann(C_1)
M_2 = mean_riemann(C_2)

p = 16

M_1_p = mean_semidefinite(C_1, p)
M_2_p = mean_semidefinite(C_2, p)

plots = [M_1, M_1_p, M_2, M_2_p]
plt.figure(figsize=(8,8))
for i,pt in enumerate(plots):
    plt.subplot(2,2,i+1)
    plt.imshow(pt)
    
#%%

def distance_subspace(UA, UB):
    M = np.dot(UA.T, UB)
    U,S,V = np.linalg.svd(M)
    S = np.clip(S, 0, 1)
    Theta = np.arccos(S)
    return np.linalg.norm(Theta)

def distance_lowrank(A, B, p, alpha=100):    
    RA,UA = eigendecomp_rank(A, p)
    RB,UB = eigendecomp_rank(B, p)    
    dist_r = distance_riemann(RA, RB)           
#    dist_l = distance_subspace(UA, UB)**2    
#    k = alpha*dist_l/dist_r    
#    return dist_l + k * dist_r
    return dist_r

from functools import partial

def acc(covs, y, means, distance):

    M_1, M_2 = means           
    preds = []
    for Ci in covs:    
        Ci = covs[i]
        d1 = distance(Ci, M_1)
        d2 = distance(Ci, M_2)
        if d1 < d2:
            preds.append(2)
        else:
            preds.append(3)
    preds = np.array(preds)
    acc = np.sum(preds == y)/(1.0*len(y))
    
    return acc

means = [M_1_p, M_2_p]
cov_  = [lowrank_approx(Ci, p) for Ci in cov]                
print acc(cov_, y, means, partial(distance_lowrank, p=p))

means = [M_1, M_2]
print acc(cov, y, means, distance_riemann)















