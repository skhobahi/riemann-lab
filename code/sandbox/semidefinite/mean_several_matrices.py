#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:18:24 2017

@author: coelhorp
"""

import numpy as np
from numpy.linalg import eig
from pyriemann.utils.mean import mean_riemann

import sys
sys.path.append('../../')

def gen_psd(n, eff=None, tol=1e-6):

    if eff is None:
        eff = n
        
    S = np.random.randn(n, n)
    S = S+S.T
    
    _,v = np.linalg.eig(S)
    w = 1.0 + 5*np.random.rand(n)
    w[eff:] = tol*np.random.rand(n-eff)
    
    A = np.dot(v, np.dot(np.diag(w), v.T))  
    
    return A+A.T
    
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
    
N = 2    
n = 4    
p = 4    

A = [gen_psd(n, p) for _ in range(N)]     
avg = mean_semidefinite(A, p)   
     
w,v = eig(avg)
print avg
print ''
print np.sort(w)

print ''
avg = mean_riemann(np.stack(A))
w,v = eig(avg)
print avg
print ''
print np.sort(w)








