#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:37:39 2019

@author: avsthiago
"""

import cv2
import numpy as np
import random
import os
import click
from tqdm import tqdm 
from pathlib import Path

def generate_slices() -> list:
    pos_x = np.arange(0, 5785,482)
    pos_y = np.arange(0, 3857,482)
    slices = [np.s_[y[0]:y[1],x[0]:x[1]] for x in zip(pos_x, pos_x + 512) for y in zip(pos_y, pos_y + 512)]
    return slices

def add_border(img):
    original_shape = img.shape[:2]

    if original_shape != (4000, 6000):
        img = cv2.resize(img, (6000, 4000))
    
    mirrored = cv2.copyMakeBorder(img,184,184,148,148, cv2.BORDER_REFLECT)
    return mirrored

def process(path_in, path_out, im_name, slices):
    img = cv2.imread(os.path.join(path_in, im_name))
    img = add_border(img)
    
    for j, sl in enumerate(slices):
        tile_name = f"{Path(im_name).stem}_{j}.JPG"
        cv2.imwrite(os.path.join(path_out, tile_name), img[sl])
        
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

@click.command()
@click.option('--in_path', "in_path", type=click.Path(exists=True))
@click.option('--out_path', "out_path")
def main(in_path, out_path):
    in_path = "/home/avsthiago/AI/DeepBee/DeepBee-source/data/raw/DS_COMB_COMB_SEG_FULL"
    out_path = "/home/avsthiago/AI/DeepBee/DeepBee-source/data/processed/DS_COMB_COMB_SEG_FULL"
    path_images = os.path.join(in_path, "images")
    path_labels = os.path.join(in_path, "labels")
    out_images_train = os.path.join(out_path, "train", "images")
    out_labels_train = os.path.join(out_path, "train", "labels")
    out_images_test = os.path.join(out_path, "test", "images")
    out_labels_test = os.path.join(out_path, "test", "labels")
    
    image_names = os.listdir(path_images)
    train_set = image_names[int(len(image_names)*0.05):]
    test_set = image_names[:int(len(image_names)*0.05)]

    slices = generate_slices()

    for path in [out_images_test, out_images_train, out_labels_test, out_labels_train]:
        create_folder(path)
    
    print("\n\nExtracting training set...")
    for i in tqdm(train_set):
        process(path_images, out_images_train, i, slices)
        process(path_labels, out_labels_train, i, slices)
    print("\nExtracting test set...")
    for i in tqdm(test_set):
        process(path_images, out_images_test, i, slices)
        process(path_labels, out_labels_test, i, slices)

if __name__ == "__main__":
    main()