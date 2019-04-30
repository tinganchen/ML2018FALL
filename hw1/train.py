#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
"""
import numpy as np
import pandas as pd
import sys

## Data Preprocessing

# -------- 1. Import raw datasets ----------------
raw_train_data = pd.read_csv(sys.argv[1], encoding = 'big5') # 'train.csv'

# Outliers detected
def detect_outliers(col):
    q1 = np.percentile(col, 25)
    q3 = np.percentile(col, 75)
    iqr = q3 - q1
    loc = np.where((col <= q1 - 1.5 * iqr) | (col >= q3 + 1.5 * iqr))[0]
    return loc

# -------- 2. x_train ------------------
contam_list = list(raw_train_data.iloc[0:18, 2].drop(10))
temp = np.zeros([5652, 1])

for contaminant in contam_list:
    train_data = raw_train_data[raw_train_data.iloc[:, 2] == contaminant].iloc[:, 3:]
    train_data.index = np.arange(train_data.shape[0])
    
    # Outliers replaced with mean
    train_data2 = pd.DataFrame(np.array(train_data.copy(), np.float32))
    for col in range(train_data.shape[1]):
        for month in range(12):
            tr_data = np.array(train_data2.iloc[month*20:(month+1)*20, col], np.float32)
            tr_m = np.mean(tr_data) # median
            tr_data[detect_outliers(tr_data)] = tr_m
            train_data2.iloc[month*20:(month+1)*20, col] = tr_data
            
    # Reform
    mat = np.array(train_data2, dtype = np.float32).reshape([12, -1])
    if contaminant == 'PM2.5':
        PM25_mat = mat
    x = []
    for month in mat:
        for i in range(471):
            x.append(month[i:(i+9)])
    temp = np.hstack((temp, np.array(x)))
    
x_train = pd.DataFrame(temp).iloc[:, 1:]

# -------- 3. y_train is always the 10th-hr PM2.5 value with no 'NR' --------
y_data = []
for month in PM25_mat:
    for i in range(9, len(month)):
        y_data.append(month[i])
y_train = pd.DataFrame(y_data)


## 4. Training
# ------ 1. train_validation sets -----------------------
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

# ----- 2. Training by GD -----------
def train(x_tr, y_tr, lr, lam, iterations, optimizer):
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr).reshape([-1])
    
    num_param = x_tr.shape[1] + 1 # number of parameters, bias and weights  
    b = 25 # initial bias
    np.random.seed(123)
    W_b = np.hstack((b, np.random.rand(num_param - 1))) # initial bias and weights

    x_tr2 = np.hstack((np.ones(x_tr.shape[0])[:, np.newaxis], x_tr)) # add a vector, [1., 1., ..., 1.]
    
    ## Adagrad
    if optimizer == 'Adagrad':
        sum_of_square_grads = np.zeros(num_param)
    ## Adam
    elif optimizer == 'Adam':
        m = 0
        v = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        alpha = 0.01
    
    rmse = []
    for iteration in range(iterations):
        y_hat = np.dot(x_tr2, W_b)
        residual = y_hat - y_tr
     
        ## L2-regularization
        loss_rmse = np.sqrt(np.mean(np.square(residual))) + lam * np.sum(W_b[1:] ** 2)
        rmse.append(loss_rmse)
        diff_regularization = 2 * lam * W_b
        diff_regularization[0] = 0.  # bias: not take regularization into account 
        
        ## Gradient
        grad = 2 * np.dot(residual, x_tr2) + np.sum(diff_regularization)
        
        ## Optimzers update parameters
        if optimizer == 'GradientDescent':  # Gradient Descent
            W_b -= lr * grad
        elif optimizer == 'Adagrad':  # Adagrad     
            sum_of_square_grads += grad ** 2
            rms_grad = np.sqrt(sum_of_square_grads)           
            W_b -= lr * grad / rms_grad
        elif optimizer == 'Adam':  # Adam
            t = iteration + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_bias_correct = m / (1 - beta1 ** t)
            v_bias_correct = v / (1 - beta2 ** t)
            W_b -= alpha * m_bias_correct / (np.sqrt(v_bias_correct) + epsilon)
        else:
            print('Optimizer is not found.')
            break
            
    return W_b, rmse

def compute_rmse(x_val, W_b, y_val):
    x_val = np.array(x_val)
    y_val = np.array(y_val).reshape([-1])
    x_val2 = np.hstack((np.ones(x_val.shape[0])[:, np.newaxis], x_val))
    y_val_pred = np.dot(x_val2, W_b)
    residual = y_val_pred - y_val
    rmse = np.sqrt(np.mean(np.square(residual)))
    return rmse, y_val_pred

# ---- All training set prune best parameters ---------
def fine_tune(x_tr, y_tr, init_W_b, lr, lam, iterations, optimizer):
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr).reshape([-1])
    
    num_param = x_tr.shape[1] + 1 # number of parameters, bias and weights  
    W_b = init_W_b # initial bias and weights

    x_tr2 = np.hstack((np.ones(x_tr.shape[0])[:, np.newaxis], x_tr)) # add a vector, [1., 1., ..., 1.]
    
    ## Adagrad
    if optimizer == 'Adagrad':
        sum_of_square_grads = np.zeros(num_param)
    ## Adam
    elif optimizer == 'Adam':
        m = 0
        v = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        alpha = 0.01
        
    rmse = []
    for iteration in range(iterations):
        y_hat = np.dot(x_tr2, W_b)
        residual = y_hat - y_tr
     
        ## L2-regularization
        loss_rmse = np.sqrt(np.mean(np.square(residual))) + lam * np.sum(W_b[1:] ** 2)
        rmse.append(loss_rmse)
        diff_regularization = 2 * lam * W_b
        diff_regularization[0] = 0.  # bias: not take regularization into account 
        
        ## Gradient
        grad = 2 * np.dot(residual, x_tr2) + np.sum(diff_regularization)
        
        ## Optimzers update parameters
        if optimizer == 'GradientDescent':  # Gradient Descent
            W_b -= lr * grad
        elif optimizer == 'Adagrad':  # Adagrad     
            sum_of_square_grads += grad ** 2
            rms_grad = np.sqrt(sum_of_square_grads)           
            W_b -= lr * grad / rms_grad
        elif optimizer == 'Adam':  # Adam
            t = iteration + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_bias_correct = m / (1 - beta1 ** t)
            v_bias_correct = v / (1 - beta2 ** t)
            W_b -= alpha * m_bias_correct / (np.sqrt(v_bias_correct) + epsilon)
        else:
            print('Optimizer is not found.')
            break
            
    return W_b, rmse

# -------- 5. Scale ----------------
x_train_scale = (x_train - x_train.mean(0)) / x_train.std(0)
x_train_scale.columns = np.arange(x_train_scale.shape[1])
moments = np.array(x_train.mean(0)), np.array(x_train.std(0))
# np.save('scale_moments.npy', moments)

# -------- 6. Amplify interaction terms and second-order-terms ----------------
## First select
# Condition: correlation coefficients in top 30
corr = []
for i in range(x_train_scale.shape[1]):
    corr.append(np.corrcoef(np.array(x_train_scale.iloc[:, i]), np.array(y_train).reshape([-1]))[0, 1])
feat_importance_order = np.flip(np.argsort(corr), 0)[:30]
# np.save('features_selection1.npy', feat_importance_order)

x_train_scale_vital_feat = x_train_scale.iloc[:, feat_importance_order]

x_tr_interact = np.empty([x_train_scale_vital_feat.shape[0], 1])
for i in range(x_train_scale_vital_feat.shape[1]):
    for j in range(x_train_scale_vital_feat.shape[1]):
        if i < j :
            tr_interact = np.multiply(x_train_scale_vital_feat.iloc[:, i], x_train_scale_vital_feat.iloc[:, j])
            x_tr_interact = np.hstack((x_tr_interact, np.array(tr_interact, np.float32).reshape([-1, 1])))
x_tr_interact2 = x_tr_interact[:, 1:]

x_train_scale_amp = pd.DataFrame(np.hstack((np.hstack((x_train_scale_vital_feat, x_tr_interact2)), 
                                            np.square(x_train_scale_vital_feat))))
## Second select - top 
corr2 = []
for i in range(x_train_scale_amp.shape[1]):
    corr2.append(np.corrcoef(np.array(x_train_scale_amp.iloc[:, i]), np.array(y_train).reshape([-1]))[0, 1])
feat_importance_order2 = np.flip(np.argsort(corr2), 0)[:100]
# np.save('features_selection2.npy', feat_importance_order2)

x_train_scale_amp2 = x_train_scale_amp.iloc[:, feat_importance_order2]

# ------ Split tr, val sets -------------------
x_val, y_val, x_tr, y_tr = split_into_tr_val(x_train_scale_amp2, y_train, 3)

lr = 1e-6
lam = 1
"""
W_bss = []
rmsess = []

val_losses = []
train_losses = []
for j in  range(x_train_scale_amp2.shape[1]):
    if j % 20 == 0:
        feat_select = range(j+1)
        
        W_bs = []
        rmses = []
        for i in range(3):
            W_b, rmse = train(x_tr[i].iloc[:, feat_select], y_tr[i], lr, lam, 3000, 'GradientDescent')
            W_bs.append(W_b)
            rmses.append(rmse)
        W_bss.append(W_bs)
        rmsess.append(rmses)
            
        val_loss = []
        train_loss = []
        for i in range(3):       
            val_loss.append(compute_rmse(x_val[i].iloc[:, feat_select], W_bs[i], y_val[i])[0])
            train_loss.append(compute_rmse(x_train_scale_amp2.iloc[:, feat_select], W_bs[i], y_train)[0])
        val_losses.append(val_loss)
        train_losses.append(train_loss)
    
val_losses, train_losses

iter_based_val_losses = []
iter_based_train_losses = []

for iteration in range(5000):
    if iteration % 20 == 0:
        W_b, rmse = train(x_tr[1], y_tr[1], lr, lam, iteration, 'GradientDescent')
        val_loss = compute_rmse(x_val[1], W_b, y_val[1])[0]
        train_loss = compute_rmse(x_train_scale_amp2, W_b, y_train)[0]
        iter_based_val_losses.append(val_loss)
        iter_based_train_losses.append(train_loss)

idx = np.argmin(iter_based_val_losses)
iter_based_val_losses[idx] # val_loss = 7.13002
iter_based_train_losses[idx] # train_loss = 7.0140575

best_iterations = [iteration for iteration in range(5000) if iteration % 20 == 0][idx] # 3780
"""

best_iterations = 3780
best_W_b = train(x_tr[1], y_tr[1], 
                 lr, lam, best_iterations, 'GradientDescent')[0]
# save parameters
# np.save('model.npy', best_W_b)  [7.40992]

lr_tuned = 7 * 1e-7
lam_tuned = 0.
"""
W_bs_tuned = []
train_losses_tuned = []
for iteration in range(4500, 8000):
    if iteration % 20 == 0:
        W_b_tuned = fine_tune(x_train_scale_amp2,
                              y_train, best_W_b, lr_tuned, lam_tuned, iteration, 'GradientDescent')[0]
        train_loss_tuned = compute_rmse(x_train_scale_amp2,
                                        W_b_tuned, y_train)[0]
        W_bs_tuned.append(W_b_tuned)
        train_losses_tuned.append(train_loss_tuned)
    
# [i for i in range(4500, 8000) if i % 20 == 0][np.argmin(train_losses_tuned)], np.min(train_losses_tuned) # (4960, 6.861317399448696)
"""

best_W_b_tuned = fine_tune(x_train_scale_amp2,
                           y_train, best_W_b, lr_tuned, lam_tuned, 4960, 'GradientDescent')[0]     

# save parameters
np.save('model_tuned.npy', best_W_b_tuned) # [7.40992] [Best!]


