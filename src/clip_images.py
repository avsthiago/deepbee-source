#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:22:02 2018

@author: avsthiago
"""

import cv2, os, multiprocessing
from tqdm  import tqdm

PATH = '/home/avsthiago/tese_thiago/thesis/thesis-deepbee/data/processed/human_test_set/size_220'

files = []

for path, subdirs, fls in os.walk(PATH):
    for name in fls:
        files.append(os.path.join(path, name))
    
def resize_and_save(im_name):
    size = 110
    img = cv2.imread(im_name)

    center = img.shape[0]//2
    
    min_y = center - size
    min_x = center - size
    max_y = center + size
    max_x = center + size
    roi = img[min_y:max_y,min_x:max_x]

    cv2.imwrite(im_name,roi)


pool = multiprocessing.Pool(processes=8)
    
with tqdm(total=len(files)) as t:
    for _ in pool.imap_unordered(resize_and_save, files):
        t.update(1)

pool.terminate()