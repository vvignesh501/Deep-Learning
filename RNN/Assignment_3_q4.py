from keras.models import load_model
import os
import re
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from numpy import dot

base_path = os.path.abspath('English Literature.txt')

with open(base_path, encoding='utf-8') as f:
    sample = f.read()


def format_data(input_string):
    clean_string = re.sub(r'\((\d+)\)', r'', input_string)

    clean_string = re.sub(r'\s\s', ' ', clean_string)
    return clean_string


formatted_string = format_data(sample)

# integer encode text
regular_exp = nltk.RegexpTokenizer(r"\w+")
sent = regular_exp.tokenize(formatted_string)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sent)
encoded = tokenizer.texts_to_sequences([sent])[0]

model1 = load_model('my_simple_model.h5')
model2 = load_model('my_rnn_model.h5')
model3 = load_model('my_gru_model.h5')

models = [model1, model2, model3]

model1_name = 'Simple model'
model2_name = 'RNN model'
model3_name = 'GRU model'

def get_embedding(embeddings):
    words_embeddings = {word: embeddings[index] for word, index in tokenizer.word_index.items()}
    return words_embeddings


def check_similarity(string1, string2):
    return dot(string1, string2) / (np.linalg.norm(string1) * np.linalg.norm(string2))


values = []
for i in range(len(models)):
    model = models[i]
    embeddings = model.layers[0].get_weights()[0]
    list_embedding = get_embedding(embeddings)
    string1 = 'have'
    string2 = 'had'
    a = list_embedding[string1]
    b = list_embedding[string2]
    word_similarity = check_similarity(a, b)
    values.append(word_similarity)

print("The cosine similarity for {} the two words is {}:".format(model1_name, values[0]))
print("The cosine similarity for {} the two words is {}:".format(model2_name, values[1]))
print("The cosine similarity for {} the two words is {}:".format(model3_name, values[2]))
