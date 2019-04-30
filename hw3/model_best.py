#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML hw3 train
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization,LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image  import ImageDataGenerator

import sys

# 1. Dataset
raw_train = pd.read_csv(sys.argv[1]) #'data/train.csv'

label = np.array(raw_train['label'])

imgs = []
for img in raw_train.iloc[:, 1]:
    imgs.append([int(i) for i in img.split(' ')])

# Split the training and validation set
train_size = round(len(raw_train) * 0.8)      
x_tr = np.array(imgs[:train_size])
y_tr = label[:train_size]
x_v = np.array(imgs[train_size:])
y_v = label[train_size:]

# Reshape
x_tr = x_tr.reshape(x_tr.shape[0], 48, 48, 1).astype('float32')
x_v = x_v.reshape(x_v.shape[0], 48, 48, 1).astype('float32')

# 2. Preprocessing
# (a) Scale the pixels - x
def pixel_scale(imgs):
    return np.array([i / 255 for i in imgs])

x_train = pixel_scale(x_tr)
x_val = pixel_scale(x_v)

# (b) One-hot encoding - y
def one_hot_label(label, num_label_kinds):
    label_one_hot = np.zeros(num_label_kinds, int)
    label_one_hot[label] = 1
    return label_one_hot

num_label_kinds = len(np.unique(y_tr))
y_train = np.array([one_hot_label(label, num_label_kinds) for label in y_tr])
y_val = np.array([one_hot_label(label, num_label_kinds) for label in y_v])

# (c) Data Augmentation
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)

# 3. CNN model
cnn_model = Sequential() # CNN

cnn_model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1),
                     padding = 'same', input_shape = (48, 48, 1)))
cnn_model.add(LeakyReLU(alpha=1./20))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.3))

filter_size = [64, 128, 256, 256]
for conv_layer in range(3):
    cnn_model.add(Conv2D(filters = filter_size[conv_layer], kernel_size = (3, 3), strides = (1, 1),
                         padding = 'same'))
    cnn_model.add(LeakyReLU(alpha=1./20))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())
"""
fc_node_size = [256, 128, 64, 32]
for fc_layer in range(4):
    cnn_model.add(Dense(units = fc_node_size[fc_layer], activation = 'relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
"""

cnn_model.add(Dense(units = 256))
cnn_model.add(LeakyReLU(alpha=1./20))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(units = 128))
cnn_model.add(LeakyReLU(alpha=1./20))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(units = 7, activation = 'softmax'))


## Model 1

model1 = cnn_model
optimizer1 = 'adagrad'
model1.compile(optimizer = optimizer1, loss = 'categorical_crossentropy',
               metrics = ['accuracy']) # 'adagrad'

model1.fit_generator(datagen.flow(x_train, y_train, batch_size = 50),
                     validation_data = (x_val,y_val),
                     steps_per_epoch=(len(x_train)*10 / 50), epochs = 70)

"""
model2.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), 
           epochs = 70, batch_size = 128) 

"""
model1.save('cnn_model01_img_augmentation.h5') #  (epoch = 30: 65.) (epoch = 70: )

accuracy1 = model1.evaluate(x_val, y_val, verbose = 0)
accuracy1


## Model 2
model2 = cnn_model
optimizer2 = optimizers.SGD(lr = 1e-2, momentum = 0.9)
model2.compile(optimizer = optimizer2, loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

model2.fit_generator(datagen.flow(x_train, y_train, batch_size = 128),
                     validation_data = (x_val,y_val),
                     steps_per_epoch=(len(x_train)*10 / 128), epochs = 70)
"""
model2.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), 
           epochs = 70, batch_size = 128) 

"""
model2.save('cnn_model02_img_augmentation.h5') #  (epoch = 30: 65.)(epoch = 70: 0.67372)
accuracy2 = model2.evaluate(x_val, y_val, verbose = 0)
accuracy2