#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:25:09 2017

@author: coelhorp
"""

import sys
sys.path.append('../')

import os

import cv2
import numpy as np
import scipy as sp
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.misc import imread

def distance_image(IX, IY):
    return np.linalg.norm(IX-IY)

from utilities.diffusion_map import get_diffusionEmbedding

def get_embeddings():
    
    path = '/localdata/coelhorp/video/frames/'
    filepaths = glob.glob(path + '*.jpg')
    
    print 'loading data'    
    X = []
    for filepath in filepaths:
        I = imread(filepath, flatten=True)
        I = I[::4, ::4]
        X.append(I)
    X = np.stack(X)   
    
    print 'getting the embeddings'    
    u,l = get_diffusionEmbedding(points=X, distance=distance_image)    
    
    print 'preparing the frames'
    print ''
    for ifig in tqdm(range(len(filepaths))):
        
        fig = plt.figure(figsize=(23.8,9.4), facecolor='white')    
        
        plt.subplot(1,2,1)
        for ui in u:
            plt.scatter(ui[1], ui[2], color='k', s=25)
        plt.scatter(u[ifig,1], u[ifig,2], s=160, color='r')       
        plt.xlim(-2,2)
        plt.ylim(-2,3)
        
        plt.subplot(1,2,2)
        plt.imshow(X[ifig], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        directory = './results/Video/frames'        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        figname = './results/Video/frames/frame_' + "{0:0>3}".format(ifig) + '.jpg'
        plt.savefig(figname)
        
        plt.close(fig)
        
    print ''
    
def make_video():

    print 'making video file'
    print ''
    
    path = './results/Video/frames/'
    filepaths = glob.glob(path + '*.jpg')
    
    # Determine the width and height from the first image
    frame = cv2.imread(filepaths[0])
    height, width, channels = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter('./results/Video/Video_embedding.mp4', fourcc, 20.0, (width, height))
    
    for filepath in tqdm(filepaths):
    
        frame = cv2.imread(filepath)
        out.write(frame) # Write out frame to video
    
    # Release everything if job is finished
    out.release()    
    
    print ''
    
    