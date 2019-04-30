#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML hw4 test.py
"""

import numpy as np
#import pandas as pd
import jieba
import csv
import re

from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
#from gensim.models.word2vec import LineSentence

import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import RepeatVector, Permute, Input, merge, Dense, Dropout, Flatten, BatchNormalization, SimpleRNN, Activation, LSTM, Bidirectional, TimeDistributed
from keras import optimizers
from keras.utils import np_utils

#from keras.preprocessing.sequence import TimeseriesGenerator

import keras.callbacks as callbacks

import sys

# 0. Load datasets
raw_test_x = np.array([line[1] for line in csv.reader(open(sys.argv[1], 'r', encoding='utf-8'))])[1:] # 'data/test_x.csv'

# 1. Preprocessing

jieba.load_userdict(sys.argv[2]) # 'data/dict.txt.big'


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
         output = token_jieba
    else:
         output = ['.'] # pad - if empty comment
    return output



# (a) Tokenize and remove emoji
print('Tokenize...')

# token_test_x = [tokenize(comment) for comment in raw_test_x] 
# np.save('savings/token_test_x_v3.npy', token_test_x)
token_test_x = list(np.load('token_test_x_v3.npy')) # 'savings/token_test_x_v3.npy'

# (b) Word2Vec
print('Word2Vec...')

word2vec_dim = 200

word2vec_tr_model = Word2Vec.load('word2vec_model_v3.model') # 'savings/word2vec_model_v3.model'

num_words = len(word2vec_tr_model.wv.vocab) + 1  # +1 for OOV words (out of vocabulary words)


# (c) Create embedding matrix
emb_matrix = np.zeros([num_words, word2vec_dim]) + 0.5  # first row is the embedding, [0.5, 0.5, ..., 0.5], for oov words
for i in range(num_words - 1):
    v = word2vec_tr_model.wv[word2vec_tr_model.wv.index2word[i]]
    emb_matrix[i+1] = v   # Plus 1 to reserve index 0 for OOV words

# (d) Convert words to index
test_sequences = []
for i, s in enumerate(token_test_x):
    vocab = word2vec_tr_model.wv.vocab
    toks = [vocab[w].index + 1 if w in vocab else 0 for w in s]  # Plus 1 to reserve index 1 for OOV words
    test_sequences.append(toks)
    
# (e) Padding - Pad sequence to same length
print('Padding...')
len_comment = list(map(len, test_sequences))
# np.max(len_comment) = 4156
# np.mean(len_comment) = 25.95
# np.percentile(len_comment, 90) = 59

max_len =  80

pad_test_x = pad_sequences(test_sequences, maxlen = max_len)


# 2. Load the trained model
model3 = keras.models.load_model('ensemble_lstm_4_layer_v2.h5') # 'savings/ensemble_lstm_4_layer_v2.h5'

# 3. Predict
pred_prob = model3.predict(pad_test_x)
prediction = [1 if pred > 0.5 else 0 for pred in pred_prob]

# 4. Save as .csv
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

save_csv(prediction, sys.argv[3]) # Take 3 minutes to run [0.75117]  # 'savings/ensemble_lstm_4_layer_v2.csv'


# =============================================================================
# Model: prediction
# epoch = 12 
# loss: 0.5040 - acc: 0.7519 - val_loss: 0.5053 - val_acc: 0.7537 [0.75117]
# =============================================================================
            
print('File Saved!')