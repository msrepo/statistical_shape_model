#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:54:23 2020

@author: ms
"""
import os
import numpy as np
import scipy.linalg 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from read_files import *
from my_interpolate import *

DATASET = 'HAND'

def get_centroids(points):
    """ obtain centroid of LV cardiac dataset which consists of two
    circles each containing 33 points """
    c1 = np.mean(points[:33],axis = 0)
    c2 = np.mean(points[33:],axis = 0) 
    return c1,c2

def unmake_1d(hand):
    return np.array([ [hand[i], hand[i+56]] for i in range(0,hand.shape[0]//2)])

def make_1d(hand):
    return np.concatenate((hand[:,0],hand[:,1])).reshape(-1)


def showImg(filename,show = False):
    plt.imshow(mpimg.imread(filename))
    plt.axis('off')
    if show:
        plt.show()


    
def showPoints(points,W=256,H=256, show = False,color = 'white'):
    points = np.array(points)
    plt.scatter(points[:,0]*W,points[:,1]*H,color=color,s = 1) 
    if show:
        plt.show()

def showSegImg(imgpath,points,W = None,H = None):
    if W is  None:
        W,H = getImageWH(imgpath)
    showImg(imgpath)
    if DATASET == 'HAND':
        pass
    else:

        showInterp(interp(points[:33]),W,H)
        showInterp(interp(points[33:]),W,H)
    showPoints(points,W,H,True)


def showCentroids(centroids,W=256,H=256):
    plt.scatter(centroids[:,0,0]*W,centroids[:,0,1]*H,marker = '4',color = 'black')
    plt.scatter(centroids[:,1,0]*W,centroids[:,1,1]*H,marker = '4',color = 'black')
    plt.axis('off')

def showPCAModes(mean_centre, mode ,title = None):
    mean_center_in = mean_centre.reshape(66,-1)[:33]
    mean_center_out = mean_centre.reshape(66,-1)[33:]

    ax1 = plt.subplot(1,2,1)
    showInterp(interp(mean_center_in),marker = 'r')
    showInterp(interp(mean_center_out),marker = 'r')
    showInterp(interp(mean_center_in + mode.reshape(66,-1)[:33]),marker = 'b')
    showInterp(interp(mean_center_out + mode.reshape(66,-1)[:33]),marker = 'b')

    plt.subplot(1,2,2, sharex = ax1,sharey = ax1)
    showInterp(interp(mean_center_in),marker = 'r')
    showInterp(interp(mean_center_out),marker = 'r')
    showInterp(interp(mean_center_in - mode.reshape(66,-1)[33:]),marker = 'g')
    showInterp(interp(mean_center_out - mode.reshape(66,-1)[33:]),marker = 'g')
    if title:
        plt.suptitle(title)
    
    plt.show()
    
def procrustes_hand(hands):
    np.testing.assert_equal(make_1d(unmake_1d(hands[0])),hands[0])
    normalized_hands = hands
    old_normalized_hands = hands
    
    fig = plt.figure()
    for hand in normalized_hands:
        showInterp(interp(unmake_1d(hand)))
    plt.title('Before Procrustes Alignment')
    plt.show()
    
    for count in range(5):
        mean_hand = np.mean(normalized_hands,axis = 0)
        for i,hand in enumerate(hands):
            _, mtx, disparity = scipy.spatial.procrustes(unmake_1d(mean_hand),
                                                         unmake_1d(hand))
            normalized_hands[i] = make_1d(mtx)

        
    fig = plt.figure()
    for hand in normalized_hands:
        showInterp(interp(unmake_1d(hand)))
    plt.title('After Procrustes Alignment')
    plt.show()
    
    return normalized_hands

def main():

    filepath = './ssm_datasets/hand/all/shapes'
    segmentationlist = readSegmentations(filepath,getxy)[0]
    hands = np.array(segmentationlist).T
    
    showSegImg(os.path.join(filepath,'0000.jpg'),unmake_1d(hands[0]),600,600)
    
    normalized_hands = procrustes_hand(hands)
    
    mean_normalized_hand = np.mean(normalized_hands,axis = 0)
    cov_mat = np.cov(normalized_hands.T)        
    eig_val, eig_vec = np.linalg.eigh(cov_mat)
    m = unmake_1d(mean_normalized_hand)
    for i in range(1,5):
        modeminus = unmake_1d(eig_vec[:,-i]*-3*np.sqrt(eig_val[-i]))+unmake_1d(mean_normalized_hand)
        modeplus = unmake_1d(eig_vec[:,-i]*3*np.sqrt(eig_val[-i]))+unmake_1d(mean_normalized_hand)
        fig = plt.figure(figsize =(11,3))
        ax1 = plt.subplot(131)
        showInterp(interp(modeminus),marker = 'b')   
        plt.subplot(132,sharex = ax1, sharey = ax1)
        showInterp(interp(m))
        plt.subplot(133,sharex = ax1, sharey = ax1)
        showInterp(interp(modeplus),marker = 'g')
        plt.suptitle('PCA Mode' + str(i))
        plt.show()
    

def lv_cardiac_pca():
    filepath = './ssm_datasets/lv_cardiac/data'
    segmentationlist = readSegmentations(filepath)
    
    lv_cardiac = np.array([np.array(segment) for segment in segmentationlist])
    mean_lv_cardiac = np.mean(lv_cardiac, axis = 0)
    
    showSegImg(imgname_from_segfilename(filepath,'c4480h_s1.asf'),
               lv_cardiac[0].reshape(-1,2))
    
    mean_centroids = np.array([get_centroids(mean_lv_cardiac.reshape(-1,2))])
    centroids = np.array([get_centroids(segment) for segment in segmentationlist ])
    
    
    diff1 = centroids[:,0,:] - mean_centroids[:,0,:]
    centred1 = lv_cardiac[:,:33,:] - diff1.reshape(14,1,2)
    diff2 = centroids[:,1,:] - mean_centroids[:,1,:]
    centred2 = lv_cardiac[:,33:,:] - diff2.reshape(14,1,2)
    

    
    centred = np.concatenate((centred1.reshape(14,-1),centred2.reshape(14,-1)),axis = 1)
    _cov_mat = np.cov(centred.T)
    mean_centred = np.mean(centred, axis = 0)
    eig_val, eig_vec = scipy.linalg.eigh(_cov_mat)
    for i in range(1,5):
        mode = eig_vec[:,-i] * 3 * np.sqrt(eig_val[-i])
        showPCAModes(mean_centred,mode,"PCA Major Mode "+ str(i))
        
    for c1,c2 in zip(centred1,centred2):
        showInterp(interp(c1),marker = 'b')
        showInterp(interp(c2))
    plt.title('Training Data LV Segmentation')
    plt.show()
    
    
if __name__ == '__main__':
    main()
