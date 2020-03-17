"""This code is used to read all news and their labels"""
import os
import glob

import nltk
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D


def to_categories(name, cat=["politics", "rec", "comp", "religion"]):
    for i in range(len(cat)):
        if str.find(name, cat[i]) > -1:
            return (i)
    print("Unexpected folder: " + name)  # print the folder name which does not include expected categories
    return ("wth")


def data_loader(images_dir):
    categories = os.listdir(data_path)
    news = []  # news content
    groups = []  # category which it belong to

    for cat in categories:

        # print("Category:" + cat)
        for the_new_path in glob.glob(data_path + '/' + cat + '/*'):
            news.append(open(the_new_path, encoding="ISO-8859-1", mode='r').read())
            groups.append(cat)

    return news, list(map(to_categories, groups))


data_path = "datasets/20news_subsampled"
news, groups = data_loader(data_path)
print(news, groups)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 200
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts([news])
word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

X1 = []
#X = tokenizer.texts_to_sequences([news])[0]
for i in range(len(news)):
    rem_exp = nltk.RegexpTokenizer(r"\w+")
    sentences = rem_exp.tokenize(news[i])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([sentences])
    encoded = tokenizer.texts_to_sequences([sentences])[0]
    X = [pad_sequences([encoded], maxlen=MAX_SEQUENCE_LENGTH, dtype='int32', padding='pre', truncating='pre', value=0.0)]
    X1.append(X)

X = np.asarray(X1)
X = np.reshape(X, (13108, 200))

y = to_categorical(groups, num_classes=4)
print(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))


