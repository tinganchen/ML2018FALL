#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw2 test.py
Generative model
"""

# 1. Import files
import numpy as np
import pandas as pd
import sys

raw_x_test = pd.read_csv(sys.argv[3], encoding = 'big5') # 'data/test_x.csv'

def save_csv(pred, filename):
    col = []
    for i in range(pred.shape[0]):
        col.append('id_{}'.format(i))       
    with open(filename, 'w', newline = '\n') as f:
        f.writelines('id,value\n')
        for i in range(pred.shape[0]):
            f.writelines(','.join([col[i], str(pred[i])+'\n']))

x_x_test = pd.DataFrame(np.load('preprocess_test.npy')[0])
x_x_test.columns = np.load('preprocess_test.npy')[1]

features_name = raw_x_test.columns.values
categorical_features = features_name[np.delete(np.arange(1, 11), 3)]
numeric_features = features_name[list(set(np.arange(len(features_name))) - 
                                      set(np.delete(np.arange(1, 11), 3)))]

def replace_outliers_te(df, colnames, q1s_tr, q3s_tr):
    df2 = pd.DataFrame(np.copy(df))
    df2.columns = df.columns
    for i in range(len(colnames)):
        col = np.array(df[colnames[i]])
        q1 = q1s_tr[i]
        q3 = q3s_tr[i]
        iqr = q3 - q1
        idx2small = np.where(col < q1 - 1.5 * iqr)
        idx2large = np.where(col > q3 + 1.5 * iqr)
        col[idx2small] = q1 - 1.5 * iqr
        col[idx2large] = q3 + 1.5 * iqr
        df2[colnames[i]] = col
    return df2

x_train_q1, x_train_q3 = np.load('train_q1_q3.npy')
x_test = replace_outliers_te(x_x_test, numeric_features, x_train_q1, x_train_q3)

moments = np.load('train_moments.npy')
x_test_scale = pd.DataFrame(np.copy(x_test))
x_test_scale.columns = x_test.columns
x_test_scale[numeric_features] = (x_test[numeric_features] - moments[0]) / moments[1]


# gpm_interact_pred_prob = gpm_interact.pred(x_test_scale, 1)
# gpm_interact_pred = np.array([np.int(i) for i in np.round(gpm_interact_pred_prob)])
gpm_interact_pred = np.load('gpm_interact.npy')
save_csv(gpm_interact_pred, sys.argv[4]) # [0.78060] '00002_gpm_interact_pred.csv'