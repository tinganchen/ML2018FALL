#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML hw4 train.py
"""

import numpy as np
#import pandas as pd
import jieba
import csv
import re

from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
#from gensim.models.word2vec import LineSentence

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import RepeatVector, Permute, Input, merge, Dense, Dropout, Flatten, BatchNormalization, SimpleRNN, Activation, LSTM, Bidirectional, TimeDistributed
from keras import optimizers
from keras.utils import np_utils
#from keras.optimizers import Adam
#from keras.preprocessing.sequence import TimeseriesGenerator

import keras.callbacks as callbacks

import sys


# 0. Load datasets
raw_train_x = np.array([line[1] for line in csv.reader(open(sys.argv[1], 'r', encoding='utf-8'))])[1:] # 'data/train_x.csv'
raw_train_y = np.array([line[1] for line in csv.reader(open(sys.argv[2], 'r'))])[1:].astype(np.int) # 'data/train_y.csv'

# 1. Preprocessing

jieba.load_userdict(sys.argv[4]) # 'data/dict.txt.big'

# Stop words
#stop_words = [] # size = 746 words
#with open('data/stop_words_chinese.txt', 'r', encoding = 'utf-8', newline = '\n') as f:
#    stop_words.extend(f.read().splitlines())

# Tokenize and remove emoji
def tokenize(comment):
    del_space_sentence = ''.join(comment.split(' ')) # delete space and then join
    '''
    del_emoji = re.sub("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF0-9a-zA-Z]", # delete emoji
                       "", str(del_space_sentence))
    
    token = re.sub(r'[，。、•【】～“”：；「」（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥<>/,!?･*∀-]', '', del_emoji) # delete punctuation
    '''
    token =  re.sub(r'[.]', '', del_space_sentence)           
    token_jieba = jieba.lcut(token)
    
    if token_jieba:
         # stop_rm = [w for w in token_jieba if w not in stop_words]
         # output = stop_rm
         output = token_jieba
    else:
         output = ['.'] # pad - if empty comment
    return output


# (a) Tokenize and remove emoji
print('Tokenize...')

#token_train_x = [tokenize(comment) for comment in raw_train_x] 
#np.save('savings/token_train_x_v3.npy', token_train_x)
token_train_x = list(np.load('token_train_x_v3.npy')) # 'savings/token_train_x_v3.npy'


# (b) Word2Vec
print('Word2Vec...')

word2vec_dim = 200

# word2vec_tr_model = Word2Vec(token_train_x, size = word2vec_dim, workers = 2, 
#                              min_count = 5, iter = 10, window = 3)
# Word2Vec.save(word2vec_tr_model, 'word2vec_model_v3.model') # 'savings/word2vec_model_v3.model'
word2vec_tr_model = Word2Vec.load('word2vec_model_v3.model') # 'savings/word2vec_model_v3.model'

num_words = len(word2vec_tr_model.wv.vocab) + 1  # +1 for OOV words (out of vocabulary words)


# (c) Create embedding matrix
emb_matrix = np.zeros([num_words, word2vec_dim]) + 0.5  # first row is the embedding, [0.5, 0.5, ..., 0.5], for oov words
for i in range(num_words - 1):
    v = word2vec_tr_model.wv[word2vec_tr_model.wv.index2word[i]]
    emb_matrix[i+1] = v   # Plus 1 to reserve index 0 for OOV words

# (d) Convert words to index
train_sequences = []
for i, s in enumerate(token_train_x):
    vocab = word2vec_tr_model.wv.vocab
    toks = [vocab[w].index + 1 if w in vocab else 0 for w in s]  # Plus 1 to reserve index 1 for OOV words
    train_sequences.append(toks)
    
# (e) Padding - Pad sequence to same length
print('Padding...')
len_comment = list(map(len, train_sequences))
# np.max(len_comment) = 4156
# np.mean(len_comment) = 25.95
# np.percentile(len_comment, 90) = 59

max_len =  80

pad_train_x = pad_sequences(train_sequences, maxlen = max_len)

# (f) Split train/valid set
train_x = pad_train_x[:int(len(pad_train_x)*0.8), :]
val_x = pad_train_x[int(len(pad_train_x)*0.8):, :]

train_y = raw_train_y[:int(len(pad_train_x)*0.8)]
val_y = raw_train_y[int(len(pad_train_x)*0.8):]


# 2. Model
print('Modeling...')
time_steps = max_len 
input_size = word2vec_dim

# LSTM
model3 = Sequential()
model3.add(Embedding(num_words,
                     word2vec_dim,
                     weights = [emb_matrix],
                     input_length = max_len,
                     trainable = False))
model3.add(LSTM(200, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5))
model3.add(Dense(units = 128, activation = 'relu'))
model3.add(Dropout(0.3))
model3.add(LSTM(128, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5))
model3.add(Dense(units = 64, activation = 'relu'))
model3.add(Dropout(0.3))
model3.add(LSTM(64, return_sequences = True, dropout = 0.3, recurrent_dropout = 0.3))
model3.add(Dense(units = 32, activation = 'relu'))
model3.add(LSTM(32, return_sequences = False, dropout = 0.3, recurrent_dropout = 0.3))
model3.add(Dropout(0.2))
model3.add(Dense(units = 1, activation = 'sigmoid'))

model3.compile(optimizer = 'adam', 
               loss = 'binary_crossentropy', # categorical_crossentropy
               metrics = ['accuracy']) #  # 'adagrad''adam'optimizers.SGD(lr = 0.02, momentum = 0.9)


print('Training...')

model3.fit(x = train_x, y = train_y, validation_data = (val_x, val_y), 
           epochs = 12, batch_size = 256)

print('Training finished!')
print('Saving model...')
model3.save('ensemble_lstm_4_layer_v2.h5') # 'savings/ensemble_lstm_4_layer_v2.h5'

# =============================================================================
# Model: ensemble_lstm_4_layer_v2.h5
# epoch = 12
# loss: 0.5040 - acc: 0.7519 - val_loss: 0.5053 - val_acc: 0.7537 [0.75117]
# =============================================================================
