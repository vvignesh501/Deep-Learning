from keras import losses, regularizers, optimizers, Sequential
from keras.models import Model
from keras.layers import Convolution2D, Conv2DTranspose, Input, Dense, MaxPooling2D, UpSampling2D, Conv2D, \
    ZeroPadding2D, Cropping2D, Reshape, Flatten
import matplotlib.pyplot as plt
import numpy as np
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scales the training and test data to range between 0 and 1.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


input_length = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))


model = Sequential()

# Encoder Layers
model.add(Conv2D(16, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(1, (3, 3), strides=(2,2), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.summary()

# Decoder Layers
model.add(Conv2D(6, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.summary()


model.compile(optimizer='rmsprop', loss='mean_squared_error')
history = model.fit(x_train, x_train, epochs=1, batch_size=128, validation_data=(x_test, x_test))


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
    plt.savefig("Q2.1_images.png", bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
