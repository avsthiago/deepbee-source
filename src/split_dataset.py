#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:56:16 2018

@author: avsthiago
"""

from os.path import join
import os
import random
import shutil

PCT_TRAIN = 50
PCT_VALIDATION = 50
#PCT_TEST = 5 the remaining images

PATH_IMAGES = "/home/avsthiago/tese_thiago/thesis/meu_teste/dataset/train" 
OUT_PATH_IMAGES = "/home/avsthiago/tese_thiago/thesis/meu_teste/dataset/out_train"
OUT_PATHS = ['Testing', 'Training', 'Validation']

def subfolders_names(path):
    files_and_folders = os.listdir(path)
    folders = list(filter(lambda x: os.path.isdir(join(PATH_IMAGES, x)), files_and_folders))
    return folders

def extract_sample(population, k):
    random.seed(3)
    subset = random.sample(population, k)
    population = list(filter(lambda x: x not in subset, population))
    return subset, population

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_main_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        for p in OUT_PATHS:
            full_out_path = os.path.join(path, p)
            os.makedirs(full_out_path)
            os.makedirs(os.path.join(full_out_path, 'Deep'))
            os.makedirs(os.path.join(full_out_path, 'Models'))

def copy_images(original_path, out_path, images):
    for i in images:
        shutil.copy(os.path.join(original_path, i), out_path)

def create_datasets():
    if os.path.exists(OUT_PATH_IMAGES):
        shutil.rmtree(OUT_PATH_IMAGES, True)
    
    folders = subfolders_names(PATH_IMAGES)
    
    create_main_path(OUT_PATH_IMAGES)
    
    for i in folders:
        full_path_name = os.path.join(PATH_IMAGES, i)
        all_images = os.listdir(full_path_name)
        tot_images = len(all_images)
        
        tot_train = PCT_TRAIN * tot_images // 100
        tot_validation = PCT_VALIDATION * tot_images // 100
                
        images_train, all_images = extract_sample(all_images, tot_train)
        # images_test will get the remaining images
        images_validation, images_test = extract_sample(all_images, tot_validation)
        
        for p in OUT_PATHS:
            final_path = join(join(join(OUT_PATH_IMAGES, p), 'Deep'), i.title())
            create_path(final_path)
            if p == 'Testing':
                copy_images(full_path_name, final_path, images_test)
            elif p == 'Training':
                copy_images(full_path_name, final_path, images_train)
            else:
                copy_images(full_path_name, final_path, images_validation)

create_datasets()
