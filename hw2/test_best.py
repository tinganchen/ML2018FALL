#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw2 test_best
Logistic Regression
"""
# 1. Import files
import numpy as np
import pandas as pd
import sys

raw_x_test = pd.read_csv(sys.argv[3], encoding = 'big5') # 'data/test_x.csv'

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

"""
 Testing
"""

def test(x_test, b, w):
    sigmoid_fun = lambda k: 1 / (1 + np.exp(-k))
    return sigmoid_fun(np.dot(x_test, w) + np.array(b))

def save_csv(pred, filename):
    col = []
    for i in range(pred.shape[0]):
        col.append('id_{}'.format(i))       
    with open(filename, 'w', newline = '\n') as f:
        f.writelines('id,value\n')
        for i in range(pred.shape[0]):
            f.writelines(','.join([col[i], str(pred[i])+'\n']))


x_test_interact = pd.DataFrame(np.copy(x_test_scale))
x_test_interact.columns = x_test_scale.columns
feat_select = np.load('feat_select.npy')
for i in range(len(feat_select)):
    for j in range(len(feat_select)):
        if j > i:
            interact = x_test_scale.loc[:, feat_select[i]] * x_test_scale.loc[:, feat_select[j]]
            x_test_interact = pd.concat([x_test_interact, interact], axis = 1)
            
ft_sel = np.load('ft_sel.npy')
x_test_interact_num_ft = x_test_interact[np.hstack((numeric_features, 0))]
x_test_interact_cat_ft = x_test_interact[categorical_features]
x_test_interact2 = pd.concat([x_test_interact_cat_ft, x_test_interact_num_ft.iloc[:, ft_sel]], 1)

ft_best_b, ft_best_w = np.load('ft_best_weights.npy')
ft_interact_pred = np.array([np.int(i) for i in np.round(test(x_test_interact2, ft_best_b, ft_best_w))])
save_csv(ft_interact_pred, sys.argv[4]) # [0.81880] [Best!]