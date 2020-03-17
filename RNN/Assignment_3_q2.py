# code for loading the format for the notebook
import os

import os
import string
import re
import tensorflow as tf
from array import array

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from collections import Counter
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, SimpleRNN, TimeDistributed, Masking
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer

# path : store the current path to convert back to it later
base_path = os.path.abspath('English Literature.txt')

with open(base_path, encoding='utf-8') as f:
    raw_text = f.read()

sample = raw_text


# print("Raw sample", sample)

###Data preprocessing - All the commas and dots are seperated by a space for data preprocessing.
def format_patent(patent):
    """Add spaces around punctuation and remove references to images/citations."""

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent


f = format_patent(sample)


# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        #tokenizer.fit_on_texts([in_text])
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        print(yhat)
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat[0][-1]:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text



# integer encode text
regular_exp = nltk.RegexpTokenizer(r"\w+")
sent = regular_exp.tokenize(f)
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(sent)
encoded1 = tokenizer1.texts_to_sequences([sent])[0]

model1 = load_model('my_rnn_model.h5')

print(generate_seq(model1, tokenizer1, 15, 'First citizen', 10))
