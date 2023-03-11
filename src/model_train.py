# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:22 2018
@author: avsthiago
"""

import os
import cv2
import numpy as np
import multiprocessing
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tqdm import tqdm
import shutil
from pathlib import PurePath
import imghdr
from collections import Counter
import random 
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

PATH_PRED_CORR = os.path.join(*list(PurePath('../annotations/predictions_corrected/').parts))
PATH_PREDS = os.path.join(*list(PurePath('../annotations/predictions').parts))

PATH_IMGS = os.path.join(*list(PurePath('../original_images/').parts))
OUT_DATASET = os.path.join(*list(PurePath('../dataset_train').parts))
PATH_MODEL = os.path.join(*list(PurePath('./model/classification.h5').parts))
PATH_TRAIN = os.path.join(OUT_DATASET, 'train') 
PATH_VAL = os.path.join(OUT_DATASET, 'validation')
CONFIG_FILE = 'config.txt'
BATCH_SIZE = 50
MAX_SAMPLES_CLASS = 50000
LABELS = ['Capped', 'Eggs', 'Honey', 'Larva', 'Nectar', 'Other', 'Pollen']

def load_configs():
    global MAX_SAMPLES_CLASS, BATCH_SIZE
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            lines = file.readlines()
            # load num_samples
            try:
                mx_sam = [i for i in lines if 'MAX_SAMPLES_CLASS' in i][0]
                MAX_SAMPLES_CLASS = int(mx_sam.split(':')[-1].strip())
            except:
                print('MAX_SAMPLES_CLASS not found in the config.txt file')
            # load batch
            try:
                mx_sam = [i for i in lines if 'BATCH_SIZE' in i][0]
                BATCH_SIZE = int(mx_sam.split(':')[-1].strip())
            except:
                print('BATCH_SIZE not found in the config.txt file')


def cross_plataform_directory():
    global PATH_PRED_CORR, PATH_IMGS, OUT_DATASET, PATH_MODEL, PATH_TRAIN, \
           PATH_VAL, PATH_PREDS
    if '\\' in PATH_PRED_CORR:
        PATH_PRED_CORR += '\\'
        PATH_IMGS += '\\'
        OUT_DATASET += '\\'
        PATH_TRAIN += '\\'
        PATH_VAL += '\\'
        PATH_PREDS += '\\'
    elif '/' in PATH_PRED_CORR:
        PATH_PRED_CORR += '/'
        PATH_IMGS += '/'
        OUT_DATASET += '/'
        PATH_TRAIN += '/'
        PATH_VAL += '/'
        PATH_PREDS += '/'


def create_folder(path):
    path = os.path.join(*PurePath(path).parts[:-1])
    if not os.path.exists(path):
        os.makedirs(path)


def extract_cells(image, pts, output_size=224, mean_radius_default=32, standardize_radius=True):

    if standardize_radius:
        pts[:, 2] = output_size / mean_radius_default * pts[:, 2] / 2
        size_border = pts[:, 2].max() + 1
        pts[:,[0, 1]] = pts[:, [0, 1]] + size_border
        img_w_border = cv2.copyMakeBorder(image,size_border, size_border, 
                                          size_border, size_border, 
                                          cv2.BORDER_REFLECT)

        ROIs = [cv2.resize(img_w_border[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]], (224, 224)) for i in pts]
    
    return ROIs


def save_image(data):
    cv2.imwrite(data[1],data[0])


def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(PATH_IMGS):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(PATH_IMGS, ''))  
    return l_images


def get_images_predictions(folder_im, folder_npy, folder_npy_correct):
    images = find_image_names()
    
    im_to_npy = lambda x: os.path.join(
                            os.path.join(*PurePath(x).parts[:-1]),
                            PurePath(x).parts[-1].split('.')[:1][0] + '.npy')                        
    
    files_path = sorted([[i, os.path.join(folder_im, i),
                          os.path.join(folder_npy, im_to_npy(i)),
                          os.path.join(folder_npy_correct, im_to_npy(i))]
                        for i in images])
    
    # filter if the image exists and has an annotation
    imgs_with_det = list(filter(lambda x: os.path.isfile(x[1])
                                 and os.path.isfile(x[2]), files_path))
    
    pred_files = []
    
    for i in imgs_with_det:
        index = -1 if os.path.isfile(i[-1]) else -2        
        pth_to_npy = i[index]
        pred_files.append([i[1], pth_to_npy])
        
    return pred_files


def get_qtd_by_class(points, labels):
    sum_predictions = Counter(points)
    return [*[sum_predictions[i] for i, j  in enumerate(labels)]]


def sum_qtd_by_class(list_qtd1, list_qtd2):
    return [i+j for i, j in zip(list_qtd1, list_qtd2)]


def create_dataset():
    shutil.rmtree(os.path.abspath(OUT_DATASET), ignore_errors=True) 
    list_im_det = get_images_predictions(PATH_IMGS, PATH_PREDS, PATH_PRED_CORR)
    random.shuffle(list_im_det)
    
    qt_h_conf = []
    dict_classes = {i: np.array([], dtype=np.object) for i in LABELS}
    dict_all = np.array([])
    
    print('\nLoading detections...')
    with tqdm(total=len(list_im_det)) as t:
        for k, p in enumerate(list_im_det):
            points = np.load(p[1])[2][:,[0,1,2,4,5,6]].astype(np.int32)
            points = points[points[:,-1]==1]
            points_h_conf = points[points[:,-2]==1]
            
            if points_h_conf.size == 0:
                continue
            
            l_h_conf = get_qtd_by_class(points_h_conf[:,3], LABELS)
            qt_h_conf = sum_qtd_by_class(l_h_conf, qt_h_conf) if qt_h_conf else l_h_conf
                        
            names = np.expand_dims(np.ones(len(points_h_conf)),1).astype(np.int32) * k
            
            points_class = np.hstack((points_h_conf[:,:-2], names))
            
            dict_all = np.vstack((dict_all, points_class)) \
                                   if dict_all.size > 0 else points_class
            
            t.update(1)
    
    max_samples = min(min(qt_h_conf), MAX_SAMPLES_CLASS)
    
    for i, j in enumerate(dict_classes.items()):
        cl, arr = j
        points_class = dict_all[dict_all[:,3]==i]
        np.random.shuffle(points_class)
        dict_classes[cl] = points_class[:max_samples]
        
    print('\nExtracting detections...')
    with tqdm(total=len(list_im_det)) as t:
        for c in range(7):
            lb_class = LABELS[c]
            create_folder(os.path.join(OUT_DATASET,'train',lb_class, 'X'))
            create_folder(os.path.join(OUT_DATASET,'validation',lb_class, 'X'))
            
        for k, n in enumerate(list_im_det):
            im_name = PurePath(n[0]).stem
            ftype = PurePath(n[0]).suffix
            path_annots = n[1]
            if os.path.isfile(path_annots):
                annotation = np.concatenate([i[i[:, -1]==k] for i in \
                                             dict_classes.values() if i.size > 0])
                if annotation.size == 0:
                    continue
                
                image = cv2.imread(n[0])            
                
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                
                for c in range(7):
                    lb_class = LABELS[c]
                    cells_class = annotation[annotation[:,-2]==c]
                    train_samples = int(len(cells_class) * .8)
                    train_set = cells_class[:train_samples]
                    val_set = cells_class[train_samples:]
                    
                    # train set
                    if train_set.size > 0:
                        imgs = extract_cells(image, np.copy(train_set))
                        names = [os.path.join(OUT_DATASET, 'train', lb_class, 
                                 im_name + '_' + str(i[0]) + str(i[1]) + ftype) for i in train_set]
                        if len(names) != len(set(names)):
                            print(k)
                        pool.map(save_image, zip(imgs, names))
    
                    # validation set
                    if val_set.size > 0:
                        imgs = extract_cells(image, np.copy(val_set))
                        names = [os.path.join(OUT_DATASET, 'validation', lb_class, 
                                              im_name + '_' + str(i[0]) + str(i[1]) + ftype) for i in val_set]
                        pool.map(save_image, zip(imgs, names))
                pool.close()
            t.update(1)


def train():
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_directory(PATH_TRAIN, 
                                                        target_size=(224, 224), 
                                                        batch_size=50)
    
    val_generator = val_datagen.flow_from_directory(PATH_VAL,
                                                    target_size=(224, 224), 
                                                    batch_size=50)
    
    earlystopper = EarlyStopping(patience=3, verbose=0)
    checkpointer = ModelCheckpoint(PATH_MODEL, verbose=1, save_best_only=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=0, 
                                            factor=0.5, 
                                            min_lr=0.000001)
    
    model = load_model(PATH_MODEL)
    
    model.fit_generator(train_generator, validation_data=val_generator,
                                  epochs=20, callbacks=[earlystopper, 
                                  learning_rate_reduction, 
                                  checkpointer])
    

    print('\nRemoving dataset.')
    shutil.rmtree(os.path.abspath(OUT_DATASET), ignore_errors=True)
    print('\nDone.')


def main():
    cross_plataform_directory()
    load_configs()
    
    print('\nCreating Dataset...\n')
    create_dataset()
    
    print('\n\nTraining...\n\n')
    train()
    input('\nPress Enter to close...')


if __name__ == '__main__':
    main()
