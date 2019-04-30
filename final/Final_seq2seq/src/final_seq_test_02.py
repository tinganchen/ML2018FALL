#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML seq2seq
keras test
"""

import numpy as np
import csv

import pandas as pd
import os

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, merge, Lambda


## 0. Load data
print('Load data...')
# (a) Feature maps
test_video_ids = []
for line in csv.reader(open('all/testing_options.csv', 'r', encoding='utf-8')):
    test_video_ids.append(line[0])

test_feat_df = pd.DataFrame(np.zeros((80 * len(test_video_ids), 4096)))

for i, video_id in enumerate(test_video_ids):
    file = os.path.join('all/testing_data/feat', video_id.encode('ascii', 'ignore').decode('utf-8') + '.npy')
    test_feat_df.iloc[i*80 : (i+1)*80, :] = np.load(file)
    
# (b) Options
options = np.load('text_preprocessing_savings/options_v2.npy')

# (c) Remove the empty captions
options_rm_empty = [[option if len(option) > 0 else '_' for option in video_opt ] for video_opt in options]


## 1. Word2vec - Options
opt2idx = list(np.load('savings/opt2idx.npy'))

max_len = 10

pad_opt = []
num_opt = []
for video_opt_seq in opt2idx:
    pad_opt.extend(pad_sequences(video_opt_seq , maxlen = max_len))
    num_opt.append(len(video_opt_seq))


# (b) Feature maps - repeat the vectors such that the input shape will cater to the captions' input shape
print('Repeat feat vectors...')
test_feat_reshape = np.array(test_feat_df).reshape([-1, 80, 4096]) # shape = (1425, 80, 4096)

test_feat = []
for i, feat in enumerate(test_feat_reshape):
    for _ in range(num_opt[i]):
        test_feat.append(feat)
    
encoder_input_data = np.array(test_feat) # feature maps
decoder_input_data = np.array([i.reshape([-1, 1]) for i in pad_opt]) # capitons


captions = np.load('text_preprocessing_savings/captions.npy')
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


# 3. Testing
print('Testing...')
# (a) Load model saved already
model = keras.models.load_model('s2s.h5') # 'savings/s2s.h5'

# (b) Predict
pred_words = model.predict([encoder_input_data, decoder_input_data])

test_data = []
for data in pred_words:
    tmp = []
    for word in data:
        tmp.append(np.argmax(word))
    test_data.append(tmp)

test_caption = []
for data in test_data:
    test_caption.append(captions_vocabulary.indices_to_sequence(data))

np.save('savings/test_caption.npy', test_caption)


# 4. Output
options_one_hot = []
for options in opt2idx:
    for opt in options:
        opt_one_hot = np.zeros(num_words)
        opt_one_hot[opt] = 1
        options_one_hot.append(opt_one_hot)
options_one_hot = np.array(options_one_hot)


avg_cos_sim = []
for i, cap in enumerate(pred_words):
    cos_sim = 0.
    count = 0
    for word in cap:
        if np.argmax(word) < 4:
            continue     
        cos_sim += np.dot(word, options_one_hot[i]) / np.sqrt(np.sum(np.square(word))) / np.sqrt(np.sum(np.square(options_one_hot[i])))
        count += 1
    avg_cos_sim.append(cos_sim / (count + 1e-9))


np_avg_cos_sim = np.array(avg_cos_sim).reshape([500, 5])

ans = []
for video in np_avg_cos_sim:
    ans.append(np.argmax(video))

'''options distribution
np.unique(ans, return_counts = True)[1] / np.sum(np.unique(ans, return_counts = True)[1])
'''

def save_csv(pred, filename):
    """
    col = []
    for i in range(len(pred)):
        col.append('id_{}'.format(i)) 
    """
    with open(filename, 'w', newline = '\n') as f:
        f.writelines('id,Ans\n')
        for i in range(len(pred)):
            f.writelines(','.join([str(i + 1), str(pred[i])+'\n']))

save_csv(ans, 'savings/seq2seq_output.csv') # 'savings/s2s.h5'

print('The output file is saved. \nPlease check it in the path, "savings/seq2seq_output.csv".')

# Model: s2s.h5
# epoch = 10
# loss: 1.7561 - acc: 0.6787 - val_loss: 1.8020 - val_acc: 0.6845 [0.4540]


