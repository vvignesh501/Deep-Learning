from keras import Input
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as tf

decoder = load_model('Q4_dec_model.h5')
decoder.summary()

plt.figure(figsize=(18, 4))
num_images = 10
np.random.seed(42)
# encoded_input = Input(shape=(2,))
#random_test_images = np.asarray([np.random.uniform(0, 1.0, size=2), np.random.uniform(0, 1.0, size=2)])
random_test_images = np.asarray([np.random.normal(0, 1.0, 10), np.random.normal(0, 1.0, 10)])
random_test_images = random_test_images.reshape((10, 2))
decoder_imgs = decoder.predict(random_test_images)
#check = decoder_imgs[0].reshape(28,28)

for i, image_idx in enumerate(random_test_images):
    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoder_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
