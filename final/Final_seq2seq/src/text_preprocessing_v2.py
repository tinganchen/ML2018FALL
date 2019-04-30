#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML final
Text Preprocessing
(Captions and Options)
"""

import numpy as np
import json
import csv

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import enchant
from autocorrect import spell as spell
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('wordnet')


# 0. Functions
# (a) Tokenization
def tokenize(alist):
    letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",          # Replace all non-letters with spaces
                          str(alist))
    token = re.sub(r'[)!;/.?:-]', ' ', letters_only)
    token = word_tokenize(token.lower())
    return token

def qc_token(q_or_c):
    qcs = []
    for i in q_or_c:
        qcs.extend(tokenize(i))
    return np.array(qcs)

# (b) Spelling correction
def correct_spelling(sentences):
    correct_sentences = []
    
    for sentence in sentences:
        correct_sentence = []
        
        for word in sentence:
            correct_sentence.append(spell(word))
            
        correct_sentences.append(correct_sentence)
        
    return correct_sentences
    
# (c) Remove stop words
# nltk.download('stopwords') # Download it first
stop_words = set(stopwords.words('english')) # 179 stop words provided in the package

def remove_stop_words(sentences):
    return [[word for word in sentence if word not in stop_words] for sentence in sentences]

# (d) Remove the words that don't exist (Look up the dictionary)
dictionary = enchant.Dict("en_US")

def look_up_dict(sentences):
    new_sentences = []
    
    for sentence in sentences:
        new_sentence = [word for word in sentence if dictionary.check(word)]
        new_sentences.append(new_sentence)
        
    return new_sentences

# (e) Unify the tense of verbs and single/plural
def unify_tense(word):
    word = WordNetLemmatizer().lemmatize(word,'v')
    word = WordNetLemmatizer().lemmatize(word,'n')
    if word == 'men':
        word = 'man'
    elif word == 'women':
        word = 'woman'
    return word


# 1. Captions
## Load data
with open("all/training_label.json", 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

# (a) Tokenize all captions
captions_data = []
for i in range(len(json_data)):
    captions_data.append(json_data[i]['caption'])

captions_token = [[tokenize(cap) for cap in video] for video in captions_data] 

# (b) Spelling Correction
captions_corrected = [correct_spelling(video_captions) for video_captions in captions_token]

# (c) Remove the stop words
captions_rm_stop_words = [remove_stop_words(video_captions) for video_captions in captions_corrected]

# (d) Remove the words that don't exist
captions_look_up_dict = [look_up_dict(video_captions) for video_captions in captions_rm_stop_words]

# (e) Unify the tense of verbs
captions = [[[unify_tense(word) for word in caption] for caption in video_captions] for video_captions in captions_look_up_dict]

# np.save('text_preprocessing_savings/captions.npy', captions)


# 2. Options
## Load data
test_data_opt = {}
for line in csv.reader(open('all/testing_options.csv', 'r', encoding='utf-8')):
    test_data_opt['{}'.format(line[0])] = line[1:]

options_data = []
for i in test_data_opt.values():
    options_data.append(i)

new_options_data = []
for video_opt in options_data:
    new_video_opt = []
    for i in range(len(video_opt)):
        if not video_opt[i]:
            continue
        elif video_opt[i][0] != ' ':
            new_video_opt.append(video_opt[i])
        else:
            new_video_opt[len(new_video_opt)-1] = new_video_opt[len(new_video_opt)-1] + video_opt[i]
    new_options_data.append(new_video_opt)
      

# (a) Tokenize all options
options_token = [[tokenize(opt) for opt in video] for video in new_options_data] 

# (b) Spelling Correction
options_corrected = [correct_spelling(options) for options in options_token]

# (c) Remove the stop words
options_rm_stop_words = [remove_stop_words(video_options) for video_options in options_corrected]

# (d) Remove the words that don't exist
options_look_up_dict = [look_up_dict(video_options) for video_options in options_rm_stop_words]

# (e) Unify the tense of verbs
options = [[[unify_tense(word) for word in option] for option in video_options] for video_options in options_look_up_dict]

np.save('text_preprocessing_savings/options_v2.npy', options)

