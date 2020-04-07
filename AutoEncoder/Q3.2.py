from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
epochs = 50
inChannel = 1
x, y = 28, 28
input_img = Input(shape=(x, y, inChannel))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Loads the training and test data sets (ignoring class labels)
(x_train, _), (x_test, _) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#(x_train.shape, x_test.shape)

# input dimension = 784
encoding_dim = 32

##Create model
model = Sequential()


# Encoder Layers
def encoder(model):
    model.add(Dense(128, input_shape=(784,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones'))
    return model


# Decoder Layers
def decoder(encoder):
    model.add(Dense(8, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model


encoder_model = encoder(model)
decoder_model = decoder(encoder_model)


model = decoder_model
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

#model1 = Sequential()
#model1.add(encoder_model)
#model1.add(decoder_model)

#autoencoder_model = Model(model1)
#model1.summary()

# retrieve the last layer of the autoencoder model
decoder_layer1 = model.layers[-5]
decoder_layer2 = model.layers[-4]
decoder_layer3 = model.layers[-3]
decoder_layer4 = model.layers[-2]
decoder_layer5 = model.layers[-1]


# create the decoder model
encoded_input = Input(shape=(2,))
decoder_layers = decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))
decoder = Model(input=encoded_input, output=decoder_layers)
decoder.summary()
decoder.save('Q3b_model.h5')

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = model.predict(x_test)

plt.figure(figsize=(18, 4))
num_images = 10
np.random.seed(42)
print("X test shape is:", x_test.shape[0])
random_test_images = np.random.randint(x_test.shape[0], size=num_images)


for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()