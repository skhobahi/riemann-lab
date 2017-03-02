#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:59:19 2017

@author: coelhorp
"""

import numpy as np

def make_distanceMatrix(points, distance):

    Npoints = points.shape[0]
    distmatrix = np.zeros((Npoints, Npoints))
    for ii,pi in enumerate(points):
        for jj,pj in enumerate(points):
            distmatrix[ii,jj] = distance(pi,pj)
            
    return distmatrix

def renormalize_kernel(kernel, alpha):    
    q = np.power(np.dot(kernel, np.ones(len(kernel))), alpha)
    K = np.divide(kernel, np.outer(q,q))    
    return K        

def make_kernelMatrix(distmatrix, eps, alpha=1.0):
    kernel = np.exp(-distmatrix**2/(4*eps))            
    kernel_r = renormalize_kernel(kernel, alpha)            
    return kernel_r

def make_transitionMatrix(kernel):    
    d = np.sqrt(np.dot(kernel, np.ones(len(kernel))))
    P = np.divide(kernel, np.outer(d, d))    
    return P

def get_diffusionEmbedding(points, distance, alpha=1.0, tdiff=0):
    
    d = make_distanceMatrix(points, distance)   
    K = make_kernelMatrix(distmatrix=d, eps=np.median(d)**2/2, alpha=alpha)
    P = make_transitionMatrix(K)
    u,s,v = np.linalg.svd(P)    
    
    phi = np.copy(u)
    for i in range(len(u)):
        phi[:,i] = (s[i]**tdiff)*np.divide(u[:,i], u[:,0])
    
    return phi, s

if __name__=='__main__':
    
    from numpy import sin, cos
    
    def distance_euc(x,y):
        return np.linalg.norm(x-y)    
    
    def gen_data(Npoints):
        points = np.zeros((Npoints, 3))
        p = 3
        q = 15
        r = 0.9
        theta = np.linspace(0, 2*np.pi/3, Npoints)
        points[:,0] = +r*cos(p*theta)
        points[:,1] = +r*sin(p*theta)
        points[:,2] = -r*sin(q*theta)
        
        idx = np.array(range(Npoints))
        np.random.shuffle(idx)
        points = points[idx]
        theta  = theta[idx]
        
        return points, theta    
    
    Npoints = 150
    distance = distance_euc
    
    points, theta = gen_data(Npoints)              
    u, s = get_diffusionEmbedding(points, distance, alpha=1.0)
  