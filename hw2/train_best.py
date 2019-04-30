#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw2 train_best
Logistic Regression
"""

import numpy as np
import pandas as pd
import sys

# 1. Import files
raw_x_train = pd.read_csv(sys.argv[1], encoding = 'big5') # 'data/train_x.csv'
y_train = pd.read_csv(sys.argv[2], encoding = 'big5') # 'data/train_y.csv'

# 2. Features' type
features_name = raw_x_train.columns.values
categorical_features = features_name[np.delete(np.arange(1, 11), 3)]
numeric_features = features_name[list(set(np.arange(len(features_name))) - 
                                      set(np.delete(np.arange(1, 11), 3)))]

# 3 Outlier detection
def detect_outliers(df, features):
    idices_outliers = []
    for feature in features:
        col = df[feature]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        idx_outliers = df[(col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)].index
        idices_outliers.extend(idx_outliers)
    return list(set(idices_outliers))    

"""
1. Data Preprocessing
"""
# 1. Concatenate x_train and y_train
"""
raw_x = pd.concat([raw_x_train, raw_x_test], axis = 0)
raw_x.index = np.arange(len(raw_x))
x = pd.DataFrame(np.copy(raw_x))
x.columns = raw_x.columns

# 2. Categorical features - re-categorize
# (a) Sex (1 = male; 2 = female)
# (b) Education (0 = others; 1 = graduate school; 2 = university; 3 = high school)
x[categorical_features[1]] = raw_x[categorical_features[1]].replace([4, 5, 6], 0)
# (c) Marriage status (1 = married; 2 = single; 3 = others)
x[categorical_features[2]] = raw_x[categorical_features[2]].replace(0, 3)
# (d) PAY_0 ~ PAY_6
#  0 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months;... 
#  9 = payment delay for nine months.
# Merge 4-9 to 3
for i in range(3, len(categorical_features)):
    x[categorical_features[i]] = raw_x[categorical_features[i]].replace([-2, -1], 0)
    x[categorical_features[i]] = x[categorical_features[i]].replace(np.arange(4, 9), 3)
x_x_train = x.iloc[:20000, :]
x_x_test = x.iloc[20000:, :]
x_x_test.index = raw_x_test.index
np.save('npy_file/preprocess_train.npy', (np.array(x_x_train), np.array(x_x_train.columns.values)))
np.save('npy_file/preprocess_test.npy', (np.array(x_x_test), np.array(x_x_test.columns.values)))
"""
x_x_train = pd.DataFrame(np.load('preprocess_train.npy')[0])
x_x_train.columns = np.load('preprocess_train.npy')[1]

# 3. Numeric featres - outliers processing
def replace_outliers_tr(df, colnames):
    df2 = pd.DataFrame(np.copy(df))
    df2.columns = df.columns
    q1s = []
    q3s = []
    for i in range(len(colnames)):
        col = np.array(df[colnames[i]])
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        idx2small = np.where(col < q1 - 1.5 * iqr)
        idx2large = np.where(col > q3 + 1.5 * iqr)
        col[idx2small] = q1 - 1.5 * iqr
        col[idx2large] = q3 + 1.5 * iqr
        df2[colnames[i]] = col
        q1s.append(q1)
        q3s.append(q3)
    return df2, q1s, q3s

x_train, x_train_q1, x_train_q3 = replace_outliers_tr(x_x_train, numeric_features)
np.save('train_q1_q3.npy', (x_train_q1, x_train_q3))

# 3. Numeric featres 
# 3_1. Scaling
moments = x_train[numeric_features].mean(0), x_train[numeric_features].std(0)
np.save('train_moments.npy', moments)
x_train_scale = pd.DataFrame(np.copy(x_train))
x_train_scale.columns = x_train.columns
x_train_scale[numeric_features] = (x_train[numeric_features] - moments[0]) / moments[1]


# 3_2. Wilk's Lambda, ssb/ssw
def wilks_lam(col, group): # group = y_train here
    col = np.array(col)
    group = np.array(group).T[0]
    # n = len(np.unique(group))
    mean = []
    std = []
    for i in np.unique(group):
        val = col[group == i]
        m, s = np.mean(val), np.std(val)
        mean.append(m)
        std.append(s)
    Mean = np.mean(col)
    ssw = np.sum(np.square(std))
    ssb = np.sum(np.square(mean - Mean))
    return ssb / ssw 
      
feat_wilks_lam = [wilks_lam(x_train_scale[i], y_train) for i in numeric_features]
order_feat_wilks_lam = numeric_features[np.flip(np.argsort(feat_wilks_lam), 0)]


"""
2. Validation Split
"""
def split_into_tr_val(x_train, y_train, fold = 3):
    val1_idx = np.array([i for i in range(x_train.shape[0]) if i % fold == 0])
    x_val = []
    y_val = []
    x_tr = []
    y_tr = []
    for i in range(fold):
        val_idx = val1_idx + i
        tr_idx = np.array(list(set(range(x_train.shape[0]))-set(val_idx)))
        
        x_val.append(x_train.iloc[val_idx, :])
        y_val.append(y_train.iloc[val_idx, :])
        x_tr.append(x_train.iloc[tr_idx, :])
        y_tr.append(y_train.iloc[tr_idx, :])
    return x_val, y_val, x_tr, y_tr

           
"""
3_1. Modeling
@ Logistic Model
"""
class Logistic_Regression():
    def __init__(self, w_init, b_init):
        self.b = w_init
        self.w = b_init
    
    def pred(self, x):
        sigmoid_fun = lambda k: 1 / (1 + np.exp(-k))
        return sigmoid_fun(np.dot(x, self.w) + np.array(self.b))
    
    def compute_accuracy(self, x, y):
        y_hat = np.round(self.pred(x))
        accuracy = np.mean(y_hat == y)
        return accuracy
    
    def train(self, batch_size, tr_x, tr_y, val_x, val_y, lr, epoch_size):
        train_size = tr_x.shape[0]           
     
        bs = []
        ws = []
        train_accs = []
        val_accs = []
        for epoch in range(epoch_size):
            for batch in range(int(train_size / batch_size)):
                batch_mask = np.arange(batch * batch_size, (batch + 1) * batch_size)
                batch_x = tr_x.iloc[batch_mask, :]
                batch_y = tr_y.iloc[batch_mask, :]
                
                y_hat = self.pred(batch_x)
                residual = batch_y - y_hat
                grad_b = -np.sum(residual) / batch_size
                grad_w = -np.dot(batch_x.T, residual) / batch_size
                self.b -= lr * grad_b
                self.w -= lr * grad_w
            bs.append(self.b)
            ws.append(self.w)
            
            train_acc = self.compute_accuracy(batch_x, batch_y)
            train_accs.append(train_acc)
            val_acc = self.compute_accuracy(val_x, val_y)
            val_accs.append(val_acc)
            
        output_dict = {'biases' : bs, 
                       'weights' : ws,
                       'train_acc' : train_accs, 
                       'val_acc' : val_accs}        
        return output_dict


"""
Training
"""

x_train_interact = pd.DataFrame(np.copy(x_train_scale))
x_train_interact.columns = x_train_scale.columns

feat_select = np.hstack((categorical_features, order_feat_wilks_lam[:8]))
np.save('feat_select.npy', feat_select)
for i in range(len(feat_select)):
    for j in range(len(feat_select)):
        if j > i:
            interact = x_train_scale.loc[:, feat_select[i]] * x_train_scale.loc[:, feat_select[j]]
            x_train_interact = pd.concat([x_train_interact, interact], axis = 1)


x_train_interact_num_ft = x_train_interact[np.hstack((numeric_features, 0))]
x_train_interact_cat_ft = x_train_interact[categorical_features]
feat_interact_wilks_lam = [wilks_lam(x_train_interact_num_ft.iloc[:, i], y_train) for i in range(x_train_interact_num_ft.shape[1])]
ft_sel = [np.flip(np.argsort(feat_interact_wilks_lam), 0)][0] #[0][:80]
np.save('ft_sel.npy', ft_sel)
x_train_interact2 = pd.concat([x_train_interact_cat_ft, x_train_interact_num_ft.iloc[:, ft_sel]], 1)

x_val, y_val, x_tr, y_tr = split_into_tr_val(x_train_interact2, y_train, 5)


batch_size = 100
lr = 0.001
epoch_size = 150

param_num = x_train_interact2.shape[1] 
model_interact = Logistic_Regression(1, np.zeros([param_num, 1])) # b, w

model_interact_tr_output = model_interact.train(batch_size, x_tr[0], y_tr[0], x_val[0], y_val[0], lr, epoch_size)
# np.argmax(model_interact_tr_output['val_acc']), np.max(model_interact_tr_output['val_acc']) # (113, 0.825)
# model_interact_tr_output['train_acc'][np.argmax(model_interact_tr_output['val_acc'])] # 0.84
best_epoch = np.argmax(model_interact_tr_output['val_acc'])
best_b = model_interact_tr_output['biases'][best_epoch]
best_w = model_interact_tr_output['weights'][best_epoch]

# Fine_tuned
lr = 0.0005
epoch_size = 150

ft_model_interact = Logistic_Regression(best_b, best_w)
ft_model_interact_tr_output = ft_model_interact.train(batch_size, x_train_interact2, y_train, x_val[0], y_val[0], lr, epoch_size)
# np.argmax(ft_model_interact_tr_output['val_acc']), np.max(ft_model_interact_tr_output['val_acc']) # (115, 0.827)
# ft_model_interact_tr_output['train_acc'][np.argmax(ft_model_interact_tr_output['val_acc'])] # 0.89
ft_best_epoch = np.argmax(ft_model_interact_tr_output['val_acc'])
ft_best_b = ft_model_interact_tr_output['biases'][best_epoch]
ft_best_w = ft_model_interact_tr_output['weights'][best_epoch]
np.save('ft_best_weights.npy', (ft_best_b, ft_best_w))
# np.mean(np.round(test(x_train_interact2, ft_best_b, ft_best_w)) == y_train) # 0.8195


