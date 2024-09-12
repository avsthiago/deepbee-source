#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:55:46 2018

@author: avsthiago
"""

import os
import sys
import cv2
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

PATH_IMAGES_TRAIN = '/home/avsthiago/tese_thiago/thesis/datasets/border_segmentation/clip_dataset/out_images'
PATH_LABELS_TRAIN = '/home/avsthiago/tese_thiago/thesis/datasets/border_segmentation/clip_dataset/out_labels'

PATH_IMAGES_TEST = '/home/avsthiago/tese_thiago/Thesis Writing/images/result_segmentation/ds_test_segmentation/test_set_segmentation/images'
PATH_LABELS_TEST = '/home/avsthiago/tese_thiago/Thesis Writing/images/result_segmentation/ds_test_segmentation/test_set_segmentation/labels'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_images = sorted(os.listdir(PATH_IMAGES_TRAIN))
train_labels = sorted(os.listdir(PATH_LABELS_TRAIN))

test_images = sorted(os.listdir(PATH_IMAGES_TEST))
test_labels = sorted(os.listdir(PATH_LABELS_TEST))

X_tr = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_tr = np.zeros((len(train_labels), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_images), total=len(train_images)):
    path_img = os.path.join(PATH_IMAGES_TRAIN,id_)
    path_lbl = os.path.join(PATH_LABELS_TRAIN,id_[:-4]+'.jpg')
    
    img = cv2.imread(path_img)[:,:,:IMG_CHANNELS]
    
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True) # TODO : ver se eu mudar o tipo do X_tr
    X_tr[n] = img
   
    mask = cv2.imread(path_lbl,0)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y_tr[n] = np.expand_dims(mask, axis=-1)
    
    
# Get and resize test images
X_test = np.zeros((len(test_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_labels), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_images), total=len(test_images)):
    
    path_img = os.path.join(PATH_IMAGES_TEST,id_)
    path_lbl = os.path.join(PATH_LABELS_TEST,id_[:-4]+'.JPG')
    
    img = cv2.imread(path_img)[:,:,:IMG_CHANNELS]
    
    #cv2.imshow('img',mask )
    
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
    mask = cv2.imread(path_lbl,0)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y_test[n] = np.expand_dims(mask, axis=-1)
    
print('Done!')

# Check if training data looks all right
ix = random.randint(0, len(train_images))
imshow(X_tr[ix])
plt.show();
imshow(np.squeeze(Y_tr[ix]))
plt.show()
"""

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
"""
tbCallBack = TensorBoard(log_dir='/home/avsthiago/tese_thiago/thesis/scripts/graph', histogram_freq=0, write_graph=True, write_images=True)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()



# Fit model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)
earlystopper = EarlyStopping(patience=6, verbose=1)
checkpointer = ModelCheckpoint('border_segmentation_model_tese.h5', verbose=1, save_best_only=True)

results = model.fit(X_tr, Y_tr, validation_split=0.2, batch_size=100, epochs=60, 
                    callbacks=[earlystopper, checkpointer, tbCallBack, learning_rate_reduction])
"""
import seaborn as sns

history = results.history

epoch = [i+1 for i in list(range(23))]

history['epoch']= epoch

hue="event", style="event",

lista_metricas = []

np.savetxt('lista_metricas.csv',lista_metricas,fmt='%s',delimiter=',')
import pandas as pd

df = pd.read_csv('lista_metricas.csv')

ax = sns.lineplot(x="Epoch", y="Value", hue="Set", style="Set", markers=True, dashes=False, data=df[df['Metric']=='loss'])
ax.axvline(17,color = 'r', linestyle='dashed', linewidth=1)
ax.set_xlim(0, 23)
ax.text(17.3, 0.81, "Loss train = 0.02858", horizontalalignment='left', size='small', color='black')
ax.text(17.3, 0.73, "Loss val.  = 0.02508", horizontalalignment='left', size='small', color='black')
ax.set_ylabel('Loss')
ax.set_title('Model Loss')


ax = sns.lineplot(x="Epoch", y="Value", hue="Set", style="Set", markers=True, dashes=False, data=df[df['Metric']=='accuracy'])
ax.axvline(17,color = 'r', linestyle='dashed', linewidth=1)
ax.set_xlim(0, 23)
ax.text(17.3, 0.90, "Acc. train = 0.9892", horizontalalignment='left', size='small', color='black')
ax.text(17.3, 0.88, "Acc. val.  = 0.9932", horizontalalignment='left', size='small', color='black')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy')

#['metric','value','set','epoch']

for j, i in enumerate(history['loss']):
    ep = j+1
    lista_metricas.append(['loss',str(i),'Train',str(ep)])

for j, i in enumerate(history['val_loss']):
    ep = j+1
    lista_metricas.append(['loss',str(i),'Validation',str(ep)])
    
for j, i in enumerate(history['acc']):
    ep = j+1
    lista_metricas.append(['accuracy',str(i),'Train',str(ep)])
    
for j, i in enumerate(history['val_acc']):
    ep = j+1
    lista_metricas.append(['accuracy',str(i),'Validation',str(ep)])

"""


# Predict on train, val and test
model = load_model('border_segmentation_model_tese.h5')
preds_train = model.predict(X_tr[:int(X_tr.shape[0]*0.8)], verbose=1)
preds_val = model.predict(X_tr[int(X_tr.shape[0]*0.8):], verbose=1)
preds_test = model.predict(X_test, verbose=1)


model.evaluate(X_tr[:int(X_tr.shape[0]*0.8)],Y_tr[:int(X_tr.shape[0]*0.8)])
model.evaluate(X_tr[int(X_tr.shape[0]*0.8):], verbose=1)
model.evaluate(X_test, Y_test)



# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
    """
interseccao = 0
uniao = 0
import cv2
for i in range(len(preds_test_t)):
    interseccao += np.sum(np.squeeze(preds_test_t[i]) * np.squeeze(Y_test[i]))
    uniao += np.sum(np.bitwise_or(np.squeeze(preds_test_t[i]), np.squeeze(Y_test[i])))
    
interseccao = 0
uniao = 0
import cv2
xis = np.copy(X_tr[:int(X_tr.shape[0]*0.8)])
yis = np.copy(Y_tr[int(X_tr.shape[0]*0.8):])

for i in range(len(preds_val_t )):
    interseccao += np.sum(np.squeeze(preds_val_t[i]) * np.squeeze(yis[i]))
    uniao += np.sum(np.bitwise_or(np.squeeze(preds_val_t[i]), np.squeeze(yis[i])))
    #break
"""
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_tr[ix])
plt.show()
imshow(np.squeeze(Y_tr[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

model.evaluate(X_test, Y_test)

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_tr[int(X_tr.shape[0]*0.8):][ix],cmap='bgr')
plt.show()
imshow(np.squeeze(Y_tr[int(Y_tr.shape[0]*0.8):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]).astype(np.bool))
plt.show()


# Perform a sanity check on some random test samples
ix = random.randint(0, len(preds_test_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_test[ix]), cmap='viridis')
plt.colorbar()
plt.show()

plt.imshow(np.squeeze(preds_test[ix]))
plt.clim(0, 1)
plt.colorbar()
plt.show();


plt.imshow(np.squeeze(preds_test_t[ix]).astype(np.bool), cmap='viridis')
plt.colorbar()
plt.show()