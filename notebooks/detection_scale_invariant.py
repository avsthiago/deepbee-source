import sys
import preprocessing
import cv2
import numpy as np
import math

def detect_cells(image):
    image = preprocessing.pipeline(image)
    
    # find all cells with different radius
    all_cells = np.array([])
    for j in range(5,50, 5):
        cells = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=2, 
                                 minDist=12, param1=145, param2=55, 
                                 minRadius=j+1, maxRadius=j+5)
        
        if cells is not None:
            cells = cells[0][:,:3].astype(np.int32)
            all_cells = np.vstack((all_cells, cells)) if all_cells.size else cells 
    
    # select best radius
    if all_cells.size == 0:
        best_radius = 33 
    else:
        # count number of occurrences of each value in array
        best_radius = np.bincount(all_cells[:,-1]).argmax()

    minDist = best_radius * 2 - ((best_radius * 9/26) + 75/26)

    minRadius = best_radius - max(2, math.floor(best_radius * .1))
    maxRadius = best_radius + max(2,math.floor(best_radius * .1))
    
    # hough to find all cells 
    cells = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=3, 
                             minDist=minDist,  param1=100, param2=25, 
                             minRadius=minRadius, maxRadius=maxRadius)
    
    if cells is not None:
        cells = cells[0][:,:3].astype(np.int32)
    
    return cells