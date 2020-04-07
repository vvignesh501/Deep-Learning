from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

decoder = load_model('Q3b_dec_model.h5')
decoder.summary()


plt.figure(figsize=(18, 4))
num_images = 10
random_test_images = np.asarray([np.random.normal(0, 1.0, 10), np.random.normal(0, 1.0, 10)])
random_test_images = random_test_images.reshape((10, 2))
view_decode_imgs = decoder.predict(random_test_images)


for i in range(len(random_test_images)):
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(view_decode_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
