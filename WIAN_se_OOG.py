#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:32:20 2018

@author: wp
"""
from scipy import ndimage   

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#def reconstruct(marker,mask):
#    if np.greater(marker,mask).any() :
#        print("hello")
#    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
#    imrec =  marker.copy()
#    count = 0
#    while (1):
#        count = count+1
#        result = imrec
#        dilation= cv.morphologyEx(result, cv.MORPH_DILATE, kernel)
#        imrec = np.minimum(dilation,mask)
#        plt.imshow(result)
#        if (np.equal(imrec,result).all()):
#            print(count)
#            break
#    
#    return result

def morph_reconstruct (mask, marker):
    temp = marker
    prev = marker
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
    count = 0
    while(1):
        count = count +1
        temp= cv.morphologyEx(marker, cv.MORPH_DILATE, kernel)
        temp = np.minimum(temp,mask)
        
        if np.equal(temp,prev).all():
#           print(count)
           break
        
        prev = temp
                
    return temp

def imcomplement(src):
    return 255-src

def imregionalmax(src):
    lm = ndimage.filters.maximum_filter( src, size = (40,40))
    return (src == lm)

rgb = cv.imread('IMG_7192.JPG');
rgb = cv.resize(rgb,(640,480))
#lab = cv.cvtColor(rgb,cv.COLOR_RGB2LAB)
#L = lab[:,:,0]
#L = cv.equalizeHist(L)
#lab[:,:,0] = L
#rgb = cv.cvtCo1lor(lab,cv.COLOR_LAB2RGB)
#rgb = cv.GaussianBlur(rgb,(5,5),0)
G = cv.cvtColor(rgb,cv.COLOR_RGB2GRAY)


sobelx = cv.Sobel(G,cv.CV_64F,1,0,ksize=7)
sobely = cv.Sobel(G,cv.CV_64F,0,1,ksize=7)

grad = np.sqrt(np.square(sobelx)+np.square(sobely))
kernel = cv.getStructuringElement(cv.MORPH_RECT,(20,20))
opening = cv.morphologyEx(G, cv.MORPH_OPEN, kernel)
erode = cv.morphologyEx(G, cv.MORPH_ERODE, kernel)
obr = morph_reconstruct(G,erode)
closing = cv.morphologyEx(G,cv.MORPH_OPEN,kernel)
obrd = cv.morphologyEx(obr,cv.MORPH_DILATE,kernel)
obrcbr = morph_reconstruct(imcomplement(obr),imcomplement(obrd))
obrcbr = imcomplement(obrcbr)

plt.imshow(obrcbr, cmap = 'gray')