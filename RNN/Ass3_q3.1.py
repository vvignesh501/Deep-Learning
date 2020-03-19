import os
import re
import nltk
import numpy as np
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, SimpleRNN, GRU, Masking, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


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

sentence_tokenize = nltk.tokenize.sent_tokenize(formatted_string)
print(sentence_tokenize)

sent_len = 16
i = 0
input_X = []
output_y = []

X = []
y1 = []
padded_sent = []
for i in range(len(sentence_tokenize)):
    rem_exp = nltk.RegexpTokenizer(r"\w+")
    sentences = rem_exp.tokenize(sentence_tokenize[i])
    encoded = tokenizer.texts_to_sequences([sentences])[0]
    if (len(encoded) < sent_len) or (len(encoded) > sent_len):
        padded_sent = pad_sequences([encoded], maxlen=sent_len, dtype='int32', padding='pre', truncating='pre',
                                    value=0.0)
    input_X = padded_sent[0][0:(sent_len - 1)]
    output_y = padded_sent[0][1:sent_len]
    X.append(input_X)
    y1.append(output_y)


# integer encode text
X = np.array(X)
num_y = np.array(y1)
print(X, num_y)
max_words = 10


vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# one hot encode outputs
y = to_categorical(num_y, num_classes=vocab_size)


# define the network architecture: a embedding followed by LSTM
embedding_size = 100
rnn_size = 50
model1 = Sequential()
model1.add(Embedding(vocab_size, embedding_size, input_length=15))
model1.add(Masking(mask_value=0.0))
model1.add(GRU(rnn_size, return_sequences=True))
model1.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

# fit network
model1.fit(X, y, epochs=15, verbose=1, batch_size=1)
model1.save("my_gru_model.h5")

# generate a sequence from a language model
def results(model, tokenizer, max_length, input_vector, n_words):
    input_text = input_vector
    for _ in range(n_words):
        #tokenizer.fit_on_texts([in_text])
        encode_txt = tokenizer.texts_to_sequences([input_text])[0]
        predicted_words = model.predict_classes(pad_sequences([encode_txt], maxlen=max_length, padding='pre'))
        #reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_words[0][-1]:
                out_word = word
                break
        input_text = " ".join((input_text, out_word))
    return input_text


#model = load_model('my_gru_model.h5')
#generated_words = results(model, tokenizer, '15', 'love', 150)
#print("The generated words are:", generated_words)