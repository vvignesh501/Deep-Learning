import os
import glob

import nltk
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, SimpleRNN


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

#tokenized_sents = [nltk.word_tokenize(i) for i in news]

max_length = 200
embed_size = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(news)
#word_index = tokenizer.word_index
#encoded = tokenizer.texts_to_sequences([news])[0]
vocab_size = len(tokenizer.word_index.items()) + 1

#word_tokenize = nltk.tokenize.sent_tokenize(news)

X1 = []
# X = tokenizer.texts_to_sequences([news])[0]
for i in range(len(news)):
    #tokenizer.fit_on_texts([news[i]])
    rem_exp = nltk.RegexpTokenizer(r"\w+")
    sentences = rem_exp.tokenize(news[i])
    encoded = tokenizer.texts_to_sequences([sentences])[0]
    X = [pad_sequences([encoded], maxlen=max_length, dtype='int32', padding='pre', truncating='pre', value=0.0)]
    X1.append(X)

X = np.asarray(X1)
X = np.reshape(X, (13108, 200))

y = to_categorical(groups, num_classes=4)
print(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X.shape[1])
model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_length=200))
model.add(SpatialDropout1D(0.2))
model.add(SimpleRNN(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, verbose=1, validation_data=(X_test, Y_test))
model.save('my_text_classification_model.h5')
accuracy = model.evaluate(X_test, Y_test)
print('Testing Loss: {:0.5f}\n  Accuracy: {:0.5f}'.format(accuracy[0], accuracy[1]))


from google.colab import drive
drive.mount('/content/drive')
data_path = "/content/drive/My Drive/20Newsgroups_subsampled/20news_subsampled/"
