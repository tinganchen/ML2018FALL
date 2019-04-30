#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw1.py
"""
import numpy as np
import pandas as pd
import sys

## Data Preprocessing

# -------- 1. Import raw datasets ----------------
raw_test_data = pd.read_csv(sys.argv[1], encoding = 'big5', header = None) # 'test.csv'


# -------- 4. x_test --------------
contam_list = list(raw_test_data.iloc[0:18, 1].drop(10))
temp_te = np.zeros([260, 1])
for contaminant in contam_list:
    x_test_data = raw_test_data[raw_test_data.iloc[:, 1] == contaminant].iloc[:, 2:]
    x_test_data.index = np.arange(x_test_data.shape[0])
    x_test_data.columns = np.arange(x_test_data.shape[1])
    temp_te = np.hstack((temp_te, np.array(x_test_data, np.float32)))
x_test = pd.DataFrame(temp_te).iloc[:, 1:]
x_test.columns = np.arange(x_test.shape[1])

# -------- 5. Scale ----------------
x_train_mean, x_train_std = np.load('scale_moments.npy')
x_test_scale = (x_test - x_train_mean) / x_train_std 

feat_importance_order_get = np.load('features_selection1.npy')
x_test_scale_vital_feat = x_test_scale.iloc[:, feat_importance_order_get]

x_te_interact = np.empty([x_test_scale_vital_feat.shape[0], 1])
for i in range(x_test_scale_vital_feat.shape[1]):
    for j in range(x_test_scale_vital_feat.shape[1]):
        if i < j :
            te_interact = np.multiply(x_test_scale_vital_feat.iloc[:, i], x_test_scale_vital_feat.iloc[:, j])
            x_te_interact = np.hstack((x_te_interact, np.array(te_interact, np.float32).reshape([-1, 1])))
x_te_interact2 = x_te_interact[:, 1:]

x_test_scale_amp = pd.DataFrame(np.hstack((np.hstack((x_test_scale_vital_feat, x_te_interact2)), 
                                           np.square(x_test_scale_vital_feat))))
feat_importance_order2_get = np.load('features_selection2.npy')

x_test_scale_amp2 = x_test_scale_amp.iloc[:, feat_importance_order2_get]

# Test
# best_W_b_get = np.load('model.npy')
best_W_b_tuned_get = np.load('model_tuned.npy')
# best_pred = np.dot(x_test_scale_amp2, best_W_b_get[1:]) + best_W_b_get[0]
best_pred_tuned = np.dot(x_test_scale_amp2, best_W_b_tuned_get[1:]) + best_W_b_tuned_get[0]

def save_csv(pred, filename):
    col = []
    for i in range(pred.shape[0]):
        col.append('id_{}'.format(i))
        
    with open(filename, 'w', newline = '\n') as f:
        f.writelines('id,value\n')
        for i in range(pred.shape[0]):
            f.writelines(','.join([col[i], str(pred[i])+'\n']))  
            
# save_csv(best_pred, 'hw1_output.csv')  
save_csv(best_pred_tuned, sys.argv[2]) #'hw1_output_tuned.csv'