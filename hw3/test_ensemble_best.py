#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble
"""

import pandas as pd
import numpy as np
import keras
import sys

# 1. Dataset
raw_test = pd.read_csv(sys.argv[1]) # 'data/test.csv'

imgs_test = []
for img in raw_test.iloc[:, 1]:
    imgs_test.append([int(i) for i in img.split(' ')])

# Reshape
x_te = np.array(imgs_test)
x_te = x_te.reshape(x_te.shape[0], 48, 48, 1).astype('float32')


# 2. Preprocessing
# (a) Scale the pixels - x
def pixel_scale(imgs):
    return np.array([i / 255 for i in imgs])

x_test = pixel_scale(x_te)

# 3. Load the trained model
model1 = keras.models.load_model('cnn_model01_img_augmentation.h5')
model2 = keras.models.load_model('cnn_model02_img_augmentation.h5')
#model3 = keras.models.load_model('savings/cnn_tutorial_model.h5')

# 4. Predict
def predict(model, model_path):
    pred_prob = model.predict(x_test)
    
    prediction = []
    for img in pred_prob:
        pred = np.argsort(img)[-1]
        prediction.append(pred)
    
    return prediction

pred1 = predict(model1, 'cnn_model01_img_augmentation.h5')
pred2 = predict(model2, 'cnn_model02_img_augmentation.h5')


pred_prob1 = model1.predict(x_test)
pred_prob2 = model2.predict(x_test)


pred = []
for i in range(len(x_test)):
    pred.append(np.argmax((pred_prob1[i] + pred_prob2[i]) / 2))

# 5. Save as .csv
def save_csv(pred, filename):
    """
    col = []
    for i in range(len(pred)):
        col.append('id_{}'.format(i)) 
    """
    with open(filename, 'w', newline = '\n') as f:
        f.writelines('id,label\n')
        for i in range(len(pred)):
            f.writelines(','.join([str(i), str(pred[i])+'\n']))

save_csv(pred, sys.argv[2]) # [0.68013] # 'savings/ensemble_keras_img_augmentation.csv'
