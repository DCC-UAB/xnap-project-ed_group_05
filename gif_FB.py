# -*- coding: utf-8 -*-
"""alpha_version_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/169QqsfCcGUdHqy_AwOerpRm2zXCTCeWB
"""

from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
import imageio

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation


# Función para aplicar el proceso de coloración a un cuadro
def colorize_frame(frame):
    # Preprocesar el cuadro de imagen (por ejemplo, convertir RGB a LAB)
    gray_frame = rgb2lab(frame)[:, :, 0]
    gray_frame = resize(gray_frame, (400, 400))
    X = gray_frame.reshape(400, 400, 1)

    # Apilar el cuadro en una entrada de 3 canales (RGB)
    # X = np.repeat(X, 3, axis=2)

    # Obtener la predicción del modelo para el cuadro
    output = model.predict(np.expand_dims(X, axis=0))
    output *= 128

    # Generar la imagen final en color
    colored_frame = np.zeros((400, 400, 3))
    colored_frame[:, :, 0] = X[:, :, 0]
    colored_frame[:, :, 1:] = output[0]

    # Convertir la imagen de LAB a RGB
    colored_frame = lab2rgb(colored_frame)

    return colored_frame


# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))


# Abrir el archivo GIF
gif = Image.open('espiral.gif')
frames = []

for frame in range(gif.n_frames):
    # Obtener el cuadro actual
    gif.seek(frame)
    frame_image = gif.copy()

    # Convertir el cuadro a RGB si es necesario
    if frame_image.mode != 'RGB':
        frame_image = frame_image.convert('RGB')

    # Ajustar el tamaño del cuadro original si es necesario
    if frame_image.size != (400, 400):
        frame_image = frame_image.resize((400, 400), Image.BILINEAR)

    # Aplicar la coloración al cuadro actual
    colored_frame = colorize_frame(np.array(frame_image))

    # Agregar el cuadro coloreado a la lista de cuadros
    frames.append(colored_frame)



# Crear la figura y el eje
fig, ax = plt.subplots()
ax.axis('off')

# Crear una lista de artistas para la animación
artists = []
for frame in frames:
    artist = [ax.imshow(frame)]
    artists.append(artist)

# Crear la animación
ani = ArtistAnimation(fig, artists, interval=gif.info['duration'], blit=True)


# Guardar los cuadros generados 
imageio.mimsave('animacion.gif', frames)

# Abrir y mostrar el GIF guardado
saved_gif = Image.open('animacion.gif')
# saved_gif.show()

# Abrir y mostrar el GIF guardado utilizando plt.imshow()
plt.imshow(saved_gif)
plt.axis('off')
plt.show()
