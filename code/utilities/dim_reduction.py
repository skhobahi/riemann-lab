#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:36:59 2016

@author: coelhorp
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.utils.base     import powm, sqrtm, invsqrtm, logm
from pyriemann.utils.distance import distance_riemann, distance_euclid, distance_logeuclid
from pyriemann.utils.mean     import mean_riemann, mean_euclid

from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.solvers import ConjugateGradient, SteepestDescent, TrustRegions
from functools import partial

from random import randrange

class RDR(BaseEstimator, TransformerMixin):    
    '''Riemannian Dimension Reduction
    
    Dimension reduction respecting the riemannian geometry of the SPD manifold.
    There are basically two kinds of dimension reduction: supervised and
    unsupervised. Different metrics and strategies might be used for reducing 
    the dimension.
    
    Parameters
    ----------
    n_components: int (default: 6)
        The number of components to reduce the dataset. 
    method: string (default: harandi unsupervised)
        Which method should be used to reduce the dimension of the dataset.
        Different approaches use different cost functions and algorithms for
        solving the optimization problem. The options are:        
            - nrme                       
            - harandi-uns
            - harandi-sup (set nw and nb)              
            - minmax (set alpha)      
            - covpca 
            - bootstrap (set nmeans and npoints)
    '''
    
    def __init__(self, n_components=6, method='harandi-uns', params={}):          
        self.n_components = n_components
        self.method = method
        self.params = params
        
    def fit(self, X, y=None):        
        self._fit(X, y)
        return self

    def transform(self, X, y=None):        
        Xnew = self._transform(X)
        return Xnew
        
    def _fit(self, X, y):   
             
        methods = {
                   'nrme'        : dim_reduction_nrme_cgd,
                   'harandi-uns' : dim_reduction_harandiuns,
                   'harandi-sup' : dim_reduction_harandisup,
                   'minmax'      : dim_reduction_minmax,    
                   'covpca'      : dim_reduction_covpca,
                   'bootstrap'   : dim_reduction_bootstrap_means
                  }    
                                   
        self.projector_ = methods[self.method](X=X,
                                               P=self.n_components,
                                               labels=y,
                                               params=self.params)                                         
    
    def _transform(self, X):        
        K = X.shape[0]
        P = self.n_components
        W = self.projector_    
        Xnew = np.zeros((K, P, P))        
        for k in range(K):            
            Xnew[k, :, :] = np.dot(W.T, np.dot(X[k, :, :], W))                        
        return Xnew          
    
def dim_reduction_nrme_cgd(X, P, labels, params):
    
    K = X.shape[0]
    nc = X.shape[1]    
    
    S = np.zeros((nc,nc))
    
    for i in range(K):
        for j in range(K):   
            if i != j:                             
                Ci, Cj = X[i,:,:], X[j,:,:]            
                Sij = np.dot(invsqrtm(Ci), np.dot(Cj, invsqrtm(Ci)))                
                S = S + powm(logm(Sij), 2)
            
    l,v = np.linalg.eig(S)
    idx = l.argsort()[::-1]   
    l = l[idx]
    v = v[:, idx]

    W = v[:, :P]    

    return W       

def dim_reduction_bootstrap_means(X, P, labels, params):

    nc = X.shape[1] 
    K  = X.shape[0]
    
    nmeans  = params['nmeans']
    npoints = params['npoints']

    # calculate the means
    Xm = np.zeros((nmeans,nc,nc))
    for n,sn in enumerate(range(nmeans)):
        selectmeans = [randrange(0, K) for _ in range(npoints)]
        Xm[n] = mean_riemann(X[selectmeans])        
           
    W = dim_reduction_nrme_cgd(Xm, P, labels, params)

    return W     

def dim_reduction_harandiuns(X, P, labels, params):
    return _reduction_landmarks_unsup(X, P)
            
def dim_reduction_landmarks(X, P, labels, params):
    if labels is None:
        return _reduction_landmarks_unsup(X, P)
    else:
        return _reduction_landmarks_sup(X, P, labels)             
    
def _reduction_landmarks_unsup(X, P):
    L = [mean_riemann(X)]
    cost, egrad = _distance_to_landmarks_riemann(X, L, P)    
    W = solve_manopt(X, P, cost, egrad)
    return W

def _reduction_landmarks_sup(X, P, labels):
    Xc = [X[labels == i] for i in np.unique(labels)]
    Lc = [mean_riemann(Xi) for Xi in Xc]                  
    c = []
    g = []
    for Xc_, Lc_ in zip(Xc, Lc):
        c_, g_ = _distance_to_landmarks_riemann(Xc_, [Lc_], P)
        c.append(c_)
        g.append(g_)
        
    def cost_sum(W, c):
        cst = 0
        for c_ in c:
            cst += c_(W)
        return cst
        
    def egrad_sum(W, g):
        grd = np.zeros(W.shape)
        for g_ in g:
            grd += g_(W)
        return grd
                    
    cost  = partial(cost_sum, c=c)
    egrad = partial(egrad_sum, g=g)    
    W = solve_manopt(X, P, cost, egrad)
    return W           
    
def _distance_to_landmarks_riemann(X, L, P):

    def log(X):
        w,v = np.linalg.eig(X)
        w_ = np.diag(np.log(w))
        return np.dot(v, np.dot(w_, v.T))
    
    def cost_kj(W, Xk, Lj):
        Xk_ = np.dot(W.T, np.dot(Xk, W))
        Lj_ = np.dot(W.T, np.dot(Lj, W))    
        return -1*distance_riemann(Xk_, Lj_)**2 # becomes a maximization
        
    def cost_j(W, X, Lj):
        cost = 0
        for Xk in X:
            cost += cost_kj(W, Xk, Lj)
        return cost
    
    def cost(W, X, L):
        cost = 0
        for Lj in L:
            cost += cost_j(W, X, Lj)
        return cost
    
    def egrad_kj(W, Xk, Lj):           
    
        Lj_red = np.dot(W.T, np.dot(Lj, W))
        Lj_red_inv = np.linalg.inv(Lj_red)            
        Xk_red = np.dot(W.T, np.dot(Xk, W))
        Xk_red_inv = np.linalg.inv(Xk_red)
    
        argL = np.dot(Xk, np.dot(W, Xk_red_inv))
        argL = argL - np.dot(Lj, np.dot(W, Lj_red_inv))            
        argR = log(np.dot(Xk_red, Lj_red_inv))            
        grd  = 4*np.dot(argL, argR)
        
        grd  = np.real(grd)
        
        return -1*grd # becomes a maximization
    
    def egrad_j(W, X, Lj):
        grad = np.zeros(W.shape)
        for Xk in X:
            grad += egrad_kj(W, Xk, Lj)
        return grad
    
    def egrad(W, X, L):
        grad = np.zeros(W.shape)
        for Lj in L:
            grad += egrad_j(W, X, Lj)
        return grad  
        
    cost  = partial(cost, X=X, L=L)
    egrad = partial(egrad, X=X, L=L)            
    return cost, egrad   
    
def dim_reduction_minmax(X, P, labels, params):   
    
    def cost_distance_to_landmark(W, Xi, Lj):
        dist = 0
        for Xk in Xi:
            Xk_ = np.dot(W.T, np.dot(Xk, W))        
            Lj_ = np.dot(W.T, np.dot(Lj, W))                   
            dist = dist + distance_riemann(Xk_, Lj_)**2
        return dist   
    
    def cost_landmark(W, Xi, L):
        cst = 0
        for Lj in L:    
            cst += cost_distance_to_landmark(W, Xi, Lj)
        return cst
    
    def cost_class(W, Xi, Lw, Lb, alpha=0.2):
        cost_w = cost_landmark(W, Xi, Lw)
        cost_b = cost_landmark(W, Xi, Lb)
        return cost_w - alpha*cost_b
        
    def cost_(W, X, M, alpha=0.2):
        cost = 0
        classes = range(len(M))    
        for cl in classes:
            Lw = [M[cl]]
            Lb = [M[i] for i in range(len(M)) if i != cl]
            Xi = X[cl]
            cost += cost_class(W, Xi, Lw, Lb, alpha)        
        return cost
              
    def log(X):
        w,v = np.linalg.eig(X)
        w_ = np.diag(np.log(w))
        return np.dot(v, np.dot(w_, v.T))
    
    def egrad_dist_riemann(W, X, Y):           
    
        X_red = np.dot(W.T, np.dot(X, W))
        X_red_inv = np.linalg.inv(X_red)            
        Y_red = np.dot(W.T, np.dot(Y, W))
        Y_red_inv = np.linalg.inv(Y_red)
    
        argL = np.dot(X, np.dot(W, X_red_inv))
        argL = argL - np.dot(Y, np.dot(W, Y_red_inv))            
        argR = log(np.dot(X_red, Y_red_inv))            
        grd  = 4*np.dot(argL, argR)
        
        grd  = np.real(grd)
        
        return grd 
        
    def egrad_distance_to_landmark(W, X, Lj):
        egrad = np.zeros(W.shape)
        for Xk in X:
            egrad += egrad_dist_riemann(W, Xk, Lj)
        return egrad   
    
    def egrad_landmark(W, Xi, L):
        egrad = np.zeros(W.shape)
        for Lj in L:    
            egrad += egrad_distance_to_landmark(W, Xi, Lj)
        return egrad
    
    def egrad_class(W, Xi, Lw, Lb, alpha=0.2):
        grad_w = egrad_landmark(W, Xi, Lw)
        grad_b = egrad_landmark(W, Xi, Lb)
        return grad_w - alpha*grad_b
        
    def egrad_(W, X, M, alpha=0.2):
        grd = np.zeros(W.shape)
        classes = range(len(M))
        for cl in classes:
            Lw = [M[cl]]
            Lb = [M[i] for i in range(len(M)) if i != cl]
            Xi = X[cl]
            grd += egrad_class(W, Xi, Lw, Lb, alpha)        
        return grd    

    alpha = params['alpha']        
        
    classes = np.unique(labels)
    Xi = [X[labels == i] for i in classes]
    M  = [mean_riemann(Xi_) for Xi_ in Xi]         

    cost  = partial(cost_, X=Xi, M=M, alpha=alpha)
    egrad = partial(egrad_, X=Xi, M=M, alpha=alpha)        
    W     = solve_manopt(X, P, cost, egrad)    
    
    return W    
    
def dim_reduction_harandisup(X, P, labels, params):   
    
    def get_distmatrix(covs):
        
        Nt,Nc,Nc = covs.shape
        D = np.zeros((Nt,Nt))
        for i,covi in enumerate(covs):
            for j,covj in enumerate(covs):
                D[i,j] = distance_logeuclid(covi, covj)
                
        return D
        
    def make_withingraph(covs, labels, D, nw):
        
        Nt,Nc,Nc = covs.shape
        Gw = np.zeros((Nt,Nt))
        for i,covi in enumerate(covs):
            
            cl     = labels[i]
            dist   = D[i,:]            
            idx    = dist.argsort()      
            
            neighw = []
            for j in idx[1:]:
                
                if labels[j] == cl:
                    neighw.append(j)
                    
                if len(neighw) == nw:
                    break
                
            neighw = np.array(neighw)
                    
            Gw[i,neighw] = 1         
            Gw[neighw,i] = 1         
            
        return Gw        
        
    def make_betweengraph(covs, labels, D, nb):
        
        Nt,Nc,Nc = covs.shape
        Gb = np.zeros((Nt,Nt))
        for i,covi in enumerate(covs):
            
            cl     = labels[i]
            dist   = D[i,:]            
            idx    = dist.argsort()      
            
            neighb = []
            for j in idx[1:]:
                
                if labels[j] is not cl:
                    neighb.append(j)
                    
                if len(neighb) == nb:
                    break
                
            neighb = np.array(neighb)
                    
            Gb[i,neighb] = 1         
            Gb[neighb,i] = 1         
            
        return Gb 
        
    def make_affinitymatrix(covs, labels, nw, nb):
        
        D  = get_distmatrix(covs)
        Gw = make_withingraph(covs, labels, D, nw)    
        Gb = make_betweengraph(covs, labels, D, nb)    
        
        return Gw - Gb    
    
    def log(X):
        w,v = np.linalg.eig(X)
        w_ = np.diag(np.log(w))
        return np.dot(v, np.dot(w_, v.T))    
    
    def egrad_dist_riemann(W, X, Y):           
    
        X_red     = np.dot(W.T, np.dot(X, W))
        X_red_inv = np.linalg.inv(X_red)            
        Y_red     = np.dot(W.T, np.dot(Y, W))
        Y_red_inv = np.linalg.inv(Y_red)
    
        argL = np.dot(X, np.dot(W, X_red_inv))
        argL = argL - np.dot(Y, np.dot(W, Y_red_inv))            
        argR = log(np.dot(X_red, Y_red_inv))            
        grd  = 4*np.dot(argL, argR)
        
        grd  = np.real(grd)
        
        return grd 
    
    def egrad_(W, X, A):
        Nt,Nc,Nc = X.shape  
        g = np.zeros(W.shape)
        for i,Xi in enumerate(X):
            for j,Xj in enumerate(X[(i+1):]):
                if (np.abs(A[i,j]) > 1e-3) and (i != j):
                    g = g + A[i,j]*egrad_dist_riemann(W, Xi, Xj)    
        return g                
    
    def cost_(W, X, A):
        Nt,Nc,Nc = X.shape
        c = 0
        for i,Xi in enumerate(X):
            for j,Xj in enumerate(X[(i+1):]):
                if (np.abs(A[i,j]) > 1e-3) and (i != j):
                    Xi_ = np.dot(W.T, np.dot(Xi, W))
                    Xj_ = np.dot(W.T, np.dot(Xj, W))
                    c   = c + A[i,j]*distance_riemann(Xi_, Xj_)**2
        return c                

    nw = params['nw']
    nb = params['nb']        
                
    A     = make_affinitymatrix(X, labels, nw, nb)
    cost  = partial(cost_,  X=X, A=A)
    egrad = partial(egrad_, X=X, A=A)    
    W     = solve_manopt(X, P, cost, egrad)    
    
    return W    
    
def solve_manopt(X, d, cost, egrad):

    D = X.shape[1]    
    manifold = Grassmann(height=D, width=d) 
    problem  = Problem(manifold=manifold, 
                       cost=cost,
                       egrad=egrad,
                       verbosity=0)  
    
    solver = ConjugateGradient(mingradnorm=1e-3)    

    M = mean_riemann(X)
    w,v = np.linalg.eig(M)
    idx = w.argsort()[::-1]
    v_ = v[:,idx]
    Wo = v_[:,:d]
    W  = solver.solve(problem, x=Wo)                        
    return W    

def dim_reduction_covpca(X, P, labels, params): 
    
    Xm  = np.mean(X, axis=0)
    w,v = np.linalg.eig(Xm)
    idx = w.argsort()[::-1]
    v = v[:,idx]
    W = v[:,:P]
    
    return W      

if __name__=='__main__':    
    print 'reloading'




    
    
    
    
    
    
    
    
    
    
    
    