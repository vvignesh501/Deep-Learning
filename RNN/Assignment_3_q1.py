import os
import re

import nltk
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, SimpleRNN
from keras.preprocessing.text import Tokenizer

base_path = os.path.abspath('English Literature.txt')

with open(base_path, encoding='utf-8') as f:
    sample = f.read()


def format_data(input_string):

    clean_string = re.sub(r'\((\d+)\)', r'', input_string)

    clean_string = re.sub(r'\s\s', ' ', clean_string)
    return clean_string


formatted_string1 = format_data(sample)
print(len(formatted_string1))

regular_exp = nltk.RegexpTokenizer(r"\w+")
formatted_string = regular_exp.tokenize(formatted_string1)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([formatted_string])

encoded = tokenizer.texts_to_sequences([formatted_string])[0]
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create word -> word sequences
sequences = []
for i in range(1, len(encoded)):
    sequence = encoded[i - 1:i + 1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))

sequences = np.array(sequences)
X, y = sequences[:, 0], sequences[:, 1]

print(len(X), len(y))

# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)


#Model Architecture
embedding_size = 100
units = 500
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=1))
model.add(SimpleRNN(units=units, input_shape=(1, 100)))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, y, epochs=10, verbose=2)


def results(model, tokenizer, input_vector, n_words):
    output_vector = input_vector
    for _ in range(n_words):
        out_word = ''
        predicted_words = model.predict_classes(np.array(tokenizer.texts_to_sequences([input_vector])[0]))
        for word, index in tokenizer.word_index.items():
            if index == predicted_words:
                generated_word = word
                break
        output_vector = " ".join((output_vector, generated_word))
        input_vector = generated_word
    return output_vector


model.save('my_simple_model.h5')
model = load_model('my_simple_model.h5')
generated_words = results(model, tokenizer, 'citizen', 10)
print("The generated words are:", generated_words)
