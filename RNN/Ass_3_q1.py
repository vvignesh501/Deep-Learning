import os
import string
import re
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
from keras.layers import Embedding, LSTM, Dense, SimpleRNN
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

    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[:.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent


f = format_patent(sample)
# print("Formatted", f)

# tokenizer = Tokenizer(filters='"#$%&*+/:;<=>?@[\\]^_`{|}~\t\n')
# tokenizer.fit_on_texts([f])
# s = tokenizer.texts_to_sequences([f])[0]
# ' '.join(tokenizer.index_word[i] for i in s)
# tokenizer.word_index.keys()

# tokens = nltk.word_tokenize(f)

# print("No of token inputs", len(tokens))

# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([f])
encoded = tokenizer.texts_to_sequences([f])[0]


# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
    in_text, result = seed_text, seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = np.array(encoded)
        # predict a word in the vocabulary
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text, result = out_word, result + ' ' + out_word
    return result


# evaluate
#print(generate_seq(model1, tokenizer, 'Speak', 6))
# evaluate
model1 = load_model('my_simple_model.h5')
print(generate_seq(model1, tokenizer, 'First', 1))





