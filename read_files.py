#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:22:50 2020

@author: ms
"""
import os
import matplotlib.image as mpimg

def skip_comments(fp):
    while True:
        last_pos = fp.tell()
        line = fp.readline()
        if not line.strip().startswith('#') and  line.strip():
           break
    fp.seek(last_pos)
    
def getxy_selective(line):
    points = line.split()
    return [float(points[2]), float(points[3])]

def getxy(line):
    points = line.split()
    return [ float(point) for point in points]

def readPoints(filepath, filename,getxy=getxy):
    with open(os.path.join(filepath,filename)) as fp:
        skip_comments(fp)
        num_points = int(fp.readline())
        skip_comments(fp)
        points = [getxy(fp.readline()) for i in range(num_points)]
    return points

def imgname_from_segfilename(filepath, filename):
    return os.path.join(filepath,filename.split(sep='.')[0]+'.bmp')


def readSegmentations(filepath,getxy = getxy,extension = 'asf'):
    segmentationlist = [ readPoints(filepath,file,getxy) for file in os.listdir(filepath)
                            if file.endswith(extension)]
    return segmentationlist

def getImageWH(filename):
    img = mpimg.imread(filename)
    return img.shape