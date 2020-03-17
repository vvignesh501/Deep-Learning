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

# integer encode text
regular_exp = nltk.RegexpTokenizer(r"\w+")
sent = regular_exp.tokenize(f)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sent)
encoded = tokenizer.texts_to_sequences([sent])[0]

sentence_tokenize = nltk.tokenize.sent_tokenize(f)
print(sentence_tokenize)

sent_len = 16
i = 0
input_X = []
output_y = []

X = []
y1 = []
padded_sent = []
for i in range(len(sentence_tokenize)):
    # sentences = nltk.word_tokenize(sentence_tokenize[i])
    rem_exp = nltk.RegexpTokenizer(r"\w+")
    sentences = rem_exp.tokenize(sentence_tokenize[i])
    #tokenizer = Tokenizer(num_words=sent_len, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    #tokenizer.fit_on_texts([sentences])
    encoded = tokenizer.texts_to_sequences([sentences])[0]
    if (len(encoded) < sent_len) or (len(encoded) > sent_len):
        padded_sent = pad_sequences([encoded], maxlen=sent_len, dtype='int32', padding='pre', truncating='pre',
                                    value=0.0)
    input_X = padded_sent[0][0:(sent_len - 1)]
    output_y = padded_sent[0][1:sent_len]
    X.append(input_X)
    y1.append(output_y)
    # pad_y = [pad_sequences([output_y], maxlen=15, dtype='int32', padding='post', truncating='post', value=0.0)]
    # X.append(input_X)
    # y1.append(output_y)

# integer encode text
X = np.array(X)
num_y = np.array(y1)
print(X, num_y)
max_words = 10

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# one hot encode outputs
y = to_categorical(num_y, num_classes=vocab_size)

print(y)
print(len(y))

# define the network architecture: a embedding followed by LSTM
embedding_size = 100
rnn_size = 50
model1 = Sequential()
# model1.add(TimeDistributed(Masking(mask_value=0)))
model1.add(Embedding(vocab_size, embedding_size, input_length=15))
model1.add(Masking(mask_value=0.0))
model1.add(SimpleRNN(rnn_size, return_sequences=True))
model1.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

# fit network
model1.fit(X, y, epochs=10, verbose=2, batch_size=1)

model1.save('my_rnn_model.h5')


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


model1.save('my_rnn_model.h5')
# evaluate
model = load_model('my_rnn_model.h5')
print(generate_seq(model1, tokenizer, 15, 'First', 5))


