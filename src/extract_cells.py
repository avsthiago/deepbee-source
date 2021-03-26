#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:13:29 2018

@author: avsthiago
"""


import cv2
import numpy as np
import os
import random
import multiprocessing
import subprocess
import gc
import utils_processing as up

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

PATH = '/home/avsthiago/tese_thiago/thesis/datasets/images_and_annotations/labels_one_file/train_test_val'
OUT_PATH = '/home/avsthiago/tese_thiago/thesis/datasets/DEFAULT_224x224_dataset_no_out'
PATH_IMAGES = '/home/avsthiago/tese_thiago/thesis/datasets/images_and_annotations'

HANOT = True
#size = 224

labels = dict()

def work(data):
    #try:
    img = cv2.imread(data[-2])
    size = int(data[-1] // 2)
    for p in data[1]:
        #print(p)
        width, height = img.shape[:2]
        min_y = p[2]-size if p[2]-size >= 0 else 0
        min_x = p[1]-size if p[1]-size >= 0 else 0
        max_y = p[2]+size if p[2]+size <= width else width
        max_x = p[1]+size if p[1]+size <= height else height
        
        roi =  img[min_y:max_y,min_x:max_x]
        #print(os.path.join(data[2],str(p[0])+'.JPG'))
        
        cv2.imwrite(os.path.join(data[2],str(p[0])+'.JPG'),roi)
        gc.collect()
    return 0


for l in os.listdir(PATH):
    labels[l] = [os.path.join(PATH,l, i) for i in os.listdir(os.path.join(PATH, l))]

sz = 224

OUT_PATH = os.path.join(OUT_PATH, str(sz))

for ind, vals in labels.items():
    for c in vals:
        cl = c.split('/')[-1]
        create_path(os.path.join(OUT_PATH,ind,cl))
        
        annot = np.loadtxt(c, delimiter=',', dtype=np.object)
        annot[:,[0,1,2,5,6]] = annot[:,[0,1,2,5,6]].astype(np.int32)
        images_names = np.unique(annot[:,4])

        images_and_points = [[a, annot[annot[:,4]==a][:,[0,5,6]], os.path.join(OUT_PATH,ind,cl),os.path.join(PATH_IMAGES, a), sz] for a in images_names]
        
        for im in images_and_points:
            image = cv2.imread(im[-2])
            pt = np.c_[im[1], (np.ones(im[1].shape[0]) * 15).astype(np.int32)]
        
            blob_imgs = up.extract_circles(image, np.copy(pt[:, 1:]), 
                                           output_size=sz, 
                                           standardize_radius=False)
        
            for save in zip(blob_imgs, pt[:,0]):
                cv2.imwrite(os.path.join(im[2],str(save[1])+'.JPG'),save[0])
            
        print(ind, cl)


        #cv2.imshow('image', blob_imgs[1])
        
        
   


OUTPUT_FOLDER_NAME = 'OUT'
OUTPUT_FOLDER_PATH = os.path.join(OUT_PATH, OUTPUT_FOLDER_NAME)
PATH_LABELS = 'DEEP'
EXTENSION = 'JPG'
PATH_FINAL_LABELS = os.path.join(PATH, PATH_LABELS)
labels = os.listdir(PATH_FINAL_LABELS)
IMAGE_FILES = os.listdir(PATH)

classes = []


def create_path_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


for im in IMAGE_FILES:
    file_name = os.path.join(PATH, im)
    if os.path.isfile(file_name):
        im = im.upper()
        os.rename(file_name, os.path.join(PATH, im))

create_path_if_not_exists(OUTPUT_FOLDER_PATH)

tot_images = 0
ind = 0

for label in labels:
    with open(os.path.join(PATH_FINAL_LABELS, label), mode='r') as file:
        tot_images += len(list(file.readlines()))


for label in labels:
    with open(os.path.join(PATH_FINAL_LABELS, label), mode='r') as file:
        
        return_tuple_int = lambda x, y: np.array([int(float(x)), int(float(y))])
        
        image_name = ".".join(["".join(label.split('.')[:-1]), EXTENSION])
        image_path = os.path.join(PATH, image_name)
        img = cv2.imread(image_path)
    
        for line in file.readlines():
            ind += 1
            line = line.split()
            category = line[0]
            p1 = return_tuple_int(*line[4:6])
            p2 = return_tuple_int(*line[6:8])
            
            # TODO: fazer isso com uma normalização gaussiana
            #noise_x = random.randint(-10, 10)
            #noise_y = random.randint(-10, 10)
            
            #p1[0] += noise_x
            #p2[0] += noise_x
            #p1[1] += noise_y
            #p2[1] += noise_y
            
            max_y, max_x = img.shape[:2]
            p1[0] = np.clip(p1[0], 0, max_x)
            p2[0] = np.clip(p2[0], 0, max_x)
            p1[1] = np.clip(p1[1], 0, max_y)
            p2[1] = np.clip(p2[1], 0, max_y)
            
            category_output_path = os.path.join(OUTPUT_FOLDER_PATH, category)
            
            if category not in classes:
                create_path_if_not_exists(category_output_path)
                classes.append(category_output_path)
                
            
            out_img_name = ".".join([str(len(os.listdir(category_output_path))+1).zfill(6),
                                    EXTENSION])
    
            out_img_path = os.path.join(category_output_path, out_img_name)
            
            
            cv2.imwrite(out_img_path , img[p1[1]:p2[1], p1[0]:p2[0]])
            #cv2.waitKey(0)
        print(label, ind, tot_images)
    