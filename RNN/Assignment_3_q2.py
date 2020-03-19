import os
import re

import nltk
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, SimpleRNN
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

base_path = os.path.abspath('English Literature.txt')

with open(base_path, encoding='utf-8') as f:
    sample = f.read()


def format_data(input_string):
    clean_string = re.sub(r'\((\d+)\)', r'', input_string)

    clean_string = re.sub(r'\s\s', ' ', clean_string)
    return clean_string


formatted_string = format_data(sample)

regular_exp = nltk.RegexpTokenizer(r"\w+")
token_sent = regular_exp.tokenize(formatted_string)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(token_sent)
encoded = tokenizer.texts_to_sequences([token_sent])[0]


def results(model, tokenizer, max_length, input_vector, n_words):
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([input_vector])[0]
        encoded = pad_sequences([encoded], maxlen=max_length, dtype='int32', padding='pre')
        predicted_words = model.predict_classes(encoded)
        #reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_words[0][-1]:
                out_word = word
                break
        input_vector = " ".join((input_vector, out_word))
    return input_vector


model = load_model('my_rnn_model.h5')
generated_words = results(model, tokenizer, 15, 'Speak', 6)
print("The generated word sequences are:", generated_words)
