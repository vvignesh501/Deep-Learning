from keras import Sequential
from keras.layers import Input, Dense
from keras.models import Model
import keras
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
epochs = 50

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scales the training and test data to range between 0 and 1.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


input_length = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((len(x_train), input_length))
x_test = x_test.reshape((len(x_test), input_length))

##Create model
model = Sequential()


# Encoder Layers
def encoder(model):
    model.add(Dense(128, input_shape=(784,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
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

decoder_model.summary()
encoder_model.summary()
model.summary()

model.compile(optimizer='rmsprop', loss='mean_squared_error')

history = model.fit(x_train, x_train, epochs=1, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

#Retrieve the decoder layer from the trained model
decoder_layer1 = model.layers[-5]
decoder_layer2 = model.layers[-4]
decoder_layer3 = model.layers[-3]
decoder_layer4 = model.layers[-2]
decoder_layer5 = model.layers[-1]

# Save the decoder model
encoded_input = Input(shape=(2,))
decoder_layers = decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))
decoder = Model(input=encoded_input, output=decoder_layers)
decoder.summary()
decoder.save('Q1_dec_model.h5')

view_decode_img = model.predict(x_test)

fig = plt.figure(figsize=(18, 4))
fig.suptitle('Original Images (up) vs Decoded Images (down)', fontsize=15)
num_random_imgs = 10
random_test_images = np.random.randint(x_test.shape[0], size=num_random_imgs)

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax1, ax2 = plt.subplot(3, num_random_imgs, i + 1), plt.subplot(3, num_random_imgs, 2 * num_random_imgs + i + 1)
    ax1.imshow(x_test[image_idx].reshape(28, 28), cmap='gray')
    ax2.imshow(view_decode_img[image_idx].reshape(28, 28), cmap='gray')
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig("Q1_images.png", bbox_inches='tight')

plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
