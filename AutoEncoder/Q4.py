from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization, Lambda
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
# network parameters
X_dim = 784  # input dimension
batch_size = 64  # mini-batch size
epochs = 1
hidden_dim = 256  # hidden layer dimension
z_dim = 2  # latent dimension

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Loads the training and test data sets (ignoring class labels)
(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
# Scales the training and test data to range between 0 and 1.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_shape = (original_dim,)
encoding_dim = 32

##Create model
model = Sequential()


# Reparameterization trick
def sampling(args):
    z_mu, z_log_var = args
    eps = tf.random_normal(K.shape(z_log_var), dtype=np.float32, mean=0., stddev=1.0, name='epsilon')
    z = z_mu + K.exp(z_log_var / 2) * eps
    return z


# Encoder Layers
inputs = Input(shape=(X_dim,), name='input')
x = Dense(hidden_dim, activation='relu')(inputs)
z_mu = Dense(z_dim, name='z_mu')(x)
z_log_var = Dense(z_dim, name='z_log_var')(x)
z = Lambda(sampling, name='z')([z_mu, z_log_var])
z = BatchNormalization(beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(z)
# Instantiate encoder model
encoder = Model(inputs, [z_mu, z_log_var, z], name='vae_encoder')
encoder.summary()

# Decoder network
z_inputs = Input(shape=(z_dim,), name='z_sampling')
x = Dense(hidden_dim, activation='relu')(z_inputs)
outputs = Dense(X_dim, activation='sigmoid')(x)
# Instantiate decoder model
decoder = Model(z_inputs, outputs, name='vae_decoder')
decoder.summary()

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vanilla_vae')

# Loss function
# Reconstruction loss
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss = reconstruction_loss * X_dim
# KL Divergence
kl_loss = 1 + z_log_var - K.square(z_mu) - K.exp(z_log_var)
kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

history = vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

decoder.save('Q4_dec_model.h5')

# encoded_imgs = encoder.predict(x_test)
decoded_imgs = vae.predict(x_test)

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
