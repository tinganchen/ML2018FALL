#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML seq2seq
keras train
"""

import numpy as np
import json
import pandas as pd
import os

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, merge, Lambda, Bidirectional, Reshape
from keras.layers import Dropout, BatchNormalization

from keras.models import Model


## 0. Load data
print('Load data...')
# (a) Feature maps
with open('all/training_label.json', 'r', encoding='utf-8') as json_file:  #  
    json_data = json.load(json_file)
    
videos_id = [video['id'] for video in json_data]

train_feat_df = pd.DataFrame(np.zeros((80 * len(json_data), 4096)))

for i, video_id in enumerate(videos_id):
    file = os.path.join('all/training_data/feat', video_id + '.npy')
    train_feat_df.iloc[i*80 : (i+1)*80, :] = np.load(file)
    
# View -  train_feat_df[0]
    
# (b) Captions
captions = np.load('text_preprocessing_savings/captions.npy')

# (c) Remove the empty captions
captions = [[caption if len(caption) > 0  else '_' for caption in video_cap] for video_cap in captions]
  

## Index
'''
word2idx
idx2word
<SOS>: start of sentence
<EOS>: end of sentence
<PAD>: padding
<UNK>: unknown words
'''
class Vocabulary:

    def __init__(self):
        self.word2idx = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
        self.idx2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
        self.num_words = 4
        self.max_length = 0
        self.sentence_list = []

    def build_vocab(self, captions):
        """Construct the relation between sentences and indices"""
        for video_caps in captions:
    	    for caption in video_caps:
                self.sentence_list.extend(caption)
                if self.max_length < len(caption):
                    self.max_length = len(caption)

                for word in caption:
                    if word not in self.word2idx:
                        self.word2idx[word] = self.num_words
                        self.idx2word[self.num_words] = word
                        self.num_words += 1

    def sequence_to_indices(self, sequence, add_eos = False, add_sos = False):
        """Transform a word sequence to index sequence
            :param sequence: a sentence composed with words
            :param add_eos: if true, add the <EOS> tag at the end of given sentence
            :param add_sos: if true, add the <SOS> tag at the beginning of given sentence
        """
        index_sequence = [self.word2idx['SOS']] if add_sos else []

        for word in sequence:
            if word not in self.word2idx:
                index_sequence.append((self.word2idx['UNK']))
            else:
                index_sequence.append(self.word2idx[word])

        if add_eos:
            index_sequence.append(self.word2idx['EOS'])

        return index_sequence

    def indices_to_sequence(self, indices):
        """Transform a list of indices
            :param indices: a list
        """
        sequence = []
        for idx in indices:
            word = self.idx2word[idx]
            if word == "EOS":
                break
            else:
                sequence.append(word)
        return sequence

    def __sentence__(self):	
        sentence = 'Vocab information:\n'
        for idx, word in self.idx2word.items():
            sentence += "Word: %s Index: %d\n" % (word, idx)
        return sentence

captions_vocabulary = Vocabulary()
captions_vocabulary.build_vocab(captions)
num_words = captions_vocabulary.num_words

## 1. Word2vec - captions
def word2idx(vocab, captions):
    videos_seq = []
    for video_cap in captions:
        sequences = []
        for i, cap in enumerate(video_cap):        
            cap2idx = [vocab[w].index + 1 if w in vocab else 0 for w in cap]  # Plus 1 to reserve index 1 for OOV words
            sequences.append(cap2idx)
        videos_seq.append(sequences)
    return videos_seq

# (a) Captions
pos_cap2idx = [[captions_vocabulary.sequence_to_indices(cap) for cap in video_cap] for video_cap in captions] # view ~ pos_cap2idx[0]

# (b) Options
options = np.load('text_preprocessing_savings/options_v2.npy')

# (c) Remove the empty captions
options_rm_empty = [[option if len(option) > 0 else '_' for option in video_opt ] for video_opt in options]
opt2idx = [[captions_vocabulary.sequence_to_indices(opt) for opt in video_opt] for video_opt in options_rm_empty] # view ~ pos_cap2idx[0]
# np.save('savings/opt2idx.npy', opt2idx)


## 2. Prepare data
print('Prepare data...')
# (a) Padding the captions - Pad sequence to same length
print('Padding...')
cap_len = [] # all captions' length
for video_cap in captions:
    cap_len.extend(list(map(len, video_cap)))
# np.max(cap_len) = 24
# np.mean(cap_len) = 4
# np.percentile(cap_len, 99) = 10

max_len = 10

pad_pos_cap = []
num_pos_cap = []
for video_cap_seq in pos_cap2idx:
    pad_pos_cap.extend(pad_sequences(video_cap_seq , maxlen = max_len))
    num_pos_cap.append(len(video_cap_seq))


# (b) Feature maps - repeat the vectors such that the input shape will cater to the captions' input shape
print('Repeat feat vectors...')
train_feat_reshape = np.array(train_feat_df).reshape([-1, 80, 4096]) # shape = (1425, 80, 4096)

pos_feat = []
for i, feat in enumerate(train_feat_reshape):
    for _ in range(num_pos_cap[i]):
        pos_feat.append(feat)


# (c) dataset
num_decoder = len(pad_pos_cap) # 51408

encoder_input_data = pos_feat # feature maps
decoder_input_data = np.array([i.reshape([-1, 1]) for i in pad_pos_cap]) # capitons

decoder_target_data = np.zeros((num_decoder, 10, num_words), dtype = 'float32')
for i in range(num_decoder): # for each caption
    for j in range(10): # for each word in the caption
        wordidx = decoder_input_data[i][j]
        decoder_target_data[i, j-1, wordidx] = 1.

np.random.seed(123)
train_idx = np.random.choice(len(encoder_input_data), int(len(encoder_input_data) * 0.8))
valid_idx = list(set(np.arange(len(encoder_input_data))) - set(train_idx))

encoder_input_train = []
decoder_input_train = []
decoder_target_train = []
for i, idx in enumerate(train_idx):
    encoder_input_train.append(encoder_input_data[idx])
    decoder_input_train.append(decoder_input_data[idx])
    decoder_target_train.append(decoder_target_data[idx])

encoder_input_valid = []
decoder_input_valid = []
decoder_target_valid = []
for i, idx in enumerate(valid_idx):
    encoder_input_valid.append(encoder_input_data[idx])
    decoder_input_valid.append(decoder_input_data[idx]) 
    decoder_target_valid.append(decoder_target_data[idx])
    

# 3. Model
# Simple 
latent_dim = 256
print('Modeling...')
#  feature maps
encoder_inputs = Input(shape = (80, 4096))

encoder = LSTM(latent_dim, return_state=True, dropout = 0.3, recurrent_dropout = 0.3)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape = (10, 1))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
# cap_embed = Embedding(num_words, latent_dim, input_length = max_len, trainable = False)
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True, dropout = 0.3, recurrent_dropout = 0.3)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state = encoder_states)
decoder_dense = Dense(num_words, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


batch_size = 128
epochs = 10

def data_loader(encoder_input, decoder_input, decoder_output, batch_size):
    data_size = len(encoder_input)
    steps = int(data_size / batch_size)
    while True:
        for step in range(steps):
            encoder_input_batch = np.array(encoder_input[step * batch_size : (step + 1) * batch_size])
            decoder_input_batch = np.array(decoder_input[step * batch_size : (step + 1) * batch_size])
            decoder_output_batch = np.array(decoder_output[step * batch_size : (step + 1) * batch_size])
            yield [encoder_input_batch, decoder_input_batch], decoder_output_batch
    

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
'''
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size = batch_size,
          epochs = epochs,
          validation_split = 0.2)
'''
model_trained = model.fit_generator(generator = data_loader(encoder_input_train, decoder_input_train, decoder_target_train, batch_size),                  
                                    validation_data = ([np.array(encoder_input_valid), np.array(decoder_input_valid)], np.array(decoder_target_valid)), 
                                    steps_per_epoch = int(len(encoder_input_train) / batch_size),
                                    epochs = epochs) 
model.save('s2s.h5') # 'savings/s2s.h5'