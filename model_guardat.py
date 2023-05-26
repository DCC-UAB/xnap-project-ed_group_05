from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

# Comprobar si el modelo ya está entrenado y guardado
if os.path.exists('colorization_model.h5'):
    # Cargar el modelo entrenado
    model = load_model('colorization_model.h5')
    image = img_to_array(load_img('puestadesolniños.jpg'))
    image = np.array(image, dtype=float)

    X = rgb2lab(1.0/255*image)[:,:,0]
    Y = rgb2lab(1.0/255*image)[:,:,1:]
    Y /= 128

    # Redimensionar 'X' y 'Y' a la forma deseada
    X = resize(X, (400, 400))
    Y = resize(Y, (400, 400, 2))
    X = X.reshape(1, 400, 400, 1)
    Y = Y.reshape(1, 400, 400, 2)
else:
    # Obtener las imágenes y preprocesarlas
    image = img_to_array(load_img('globosniña.jpg'))
    image = np.array(image, dtype=float)

    X = rgb2lab(1.0/255*image)[:,:,0]
    Y = rgb2lab(1.0/255*image)[:,:,1:]
    Y /= 128

    # Redimensionar 'X' y 'Y' a la forma deseada
    X = resize(X, (400, 400))
    Y = resize(Y, (400, 400, 2))
    X = X.reshape(1, 400, 400, 1)
    Y = Y.reshape(1, 400, 400, 2)

    # Construir la red neuronal
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    # Finalizar el modelo
    model.compile(optimizer='rmsprop', loss='mse')

    history = model.fit(x=X, 
                        y=Y,
                        batch_size=1,
                        epochs=1000)

    # Guardar el modelo entrenado
    model.save('colorization_model.h5')

    # Trazar la función de pérdida (loss) en función de las épocas
    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot_guardat.png')
    plt.show()

output = model.predict(X)
output *= 128

# Guardar las imágenes en color
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result_flors209.png", lab2rgb(cur))
imsave("img_gray_version_flors209.png", rgb2gray(lab2rgb(cur)))
