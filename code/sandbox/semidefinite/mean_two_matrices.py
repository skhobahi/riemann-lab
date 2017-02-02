#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:56:20 2017

@author: coelhorp
"""

# Implementing the algorithm proposed in:    
# Bonnabel and Sepulchre - "Riemannian Metric and Geometric Mean for Positive 
# Semidefinite Matrices of Fixed Rank" (2009)

import numpy as np
from pyriemann.utils.mean import mean_riemann

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
    
def mean_semidefi(A, B, p):    

    RsqA, VA = eigendecomp_rank(A, p)
    RsqB, VB = eigendecomp_rank(B, p)
    
    OA, Sigma, OB = np.linalg.svd(np.dot(VA.T, VB))
    Sigma = np.clip(Sigma, 0, 1)
    Theta = np.arccos(Sigma)
    
    UA = np.dot(VA, OA)
    UB = np.dot(VB, OB)
    
    X = np.dot(np.eye(n) - np.dot(UA, UA.T), UB)
    sTheta = np.diag(np.sin(Theta))
    X = np.dot(X, np.linalg.pinv(sTheta))
    
    cTheta = np.diag(np.cos(Theta/2))
    Wl = np.dot(UA, cTheta)
    
    sTheta = np.diag(np.sin(Theta/2))
    Wr = np.dot(X, sTheta)
    
    W = Wl + Wr
    K = mean_riemann(np.stack([RsqA, RsqB]))
    avg = np.dot(W, np.dot(K, W.T))
    
    return avg
    
n = 4
p = 4   
    
A = gen_psd(n, p)  
B = gen_psd(n, p)
avg = mean_semidefi(A, B, p)    

print 'average with the semi-definite mean'
print avg
print ''

avg = mean_riemann(np.stack([A, B]))
print 'average with the positive definite mean'
print avg

     
     
     
     