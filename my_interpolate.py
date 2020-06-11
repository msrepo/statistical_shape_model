#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:46:34 2020

@author: ms
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



def interp(points, closed_curve = False,k_=3,s_=0):
    if closed_curve:
        points = np.vstack((points,points[0]))
    tck, u = interpolate.splprep([points[:,0],points[:,1]], k=k_,s=s_)
    if closed_curve:
        u = np.linspace(0,1,num = 150,endpoint = True)
    else:
        u = np.linspace(0,1,num = 150,endpoint = False)
    interp_inner = interpolate.splev(u, tck)
    return interp_inner

def showInterp(interp_points,W=256,H=256,marker = 'r'):
    plt.plot(interp_points[0]*W,interp_points[1]*H,marker)
    plt.axis('off')
