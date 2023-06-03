from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
#from tensorflow.keras.layers import BatchNormalization
from keras.layers import BatchNormalization

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from skimage.transform import resize


# Get images
image = img_to_array(load_img('niñosplaya.jpg'))
image = np.array(image, dtype=float)

X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128

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

optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer='rmsprop', loss='mse',run_eagerly=True)

# Split data into training and validation sets
val_split = 0.2
split_idx = int((1 - val_split) * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

history = model.fit(x=X_train,
                    y=Y_train,
                    batch_size=1,
                    epochs=1000,
                    validation_data=(X_val, Y_val))

print(model.evaluate(X_val, Y_val, batch_size=1))
output = model.predict(X_val)
output *= 128

# Output colorizations
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X_val[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result_capa_lr_val.png", lab2rgb(cur))
imsave("img_gray_capa_lr_val.png", rgb2gray(lab2rgb(cur)))

# Plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_capa_lr_val.png')
plt.show()
