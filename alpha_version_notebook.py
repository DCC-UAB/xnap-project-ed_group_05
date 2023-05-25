# -*- coding: utf-8 -*-
"""alpha_version_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/169QqsfCcGUdHqy_AwOerpRm2zXCTCeWB
"""

from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

# Get images
<<<<<<< HEAD
#<<<<<<< HEAD
#image = img_to_array(load_img('swim.jpg'))
#=======
#image = img_to_array(load_img('globosniña.jpg'))
#>>>>>>> 78f3cde86db549bc92afe8f08a5a8bfcb82d8886
image = img_to_array(load_img('0209.png'))
=======
<<<<<<< HEAD
image = img_to_array(load_img('man.jpg'))
=======
image = img_to_array(load_img('globosniña.jpg'))
>>>>>>> 78f3cde86db549bc92afe8f08a5a8bfcb82d8886
>>>>>>> 7259646f17f3f4f71ec082173efd1fa9b35a6142
image = np.array(image, dtype=float)

X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128
from skimage.transform import resize

# Redimensionar 'X' y 'Y' a la forma deseada
X = resize(X, (400, 400))
Y = resize(Y, (400, 400, 2))
X = X.reshape(1, 400, 400, 1)
Y = Y.reshape(1, 400, 400, 2)

# Building the neural network
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

# Finish model
model.compile(optimizer='rmsprop',loss='mse')

model.fit(x=X, 
    y=Y,
    batch_size=1,
    epochs=1000)

print(model.evaluate(X, Y, batch_size=1))
output = model.predict(X)
output *= 128

# Output colorizations
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result_flors209.png", lab2rgb(cur))
imsave("img_gray_version_flors209.png", rgb2gray(lab2rgb(cur)))
