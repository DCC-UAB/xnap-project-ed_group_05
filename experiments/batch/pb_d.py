import tensorflow as tf
from keras.optimizers import RMSprop
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
import os
import random
import matplotlib.pyplot as plt

import wandb 
from wandb.keras import WandbCallback
import random


# Set up GPU device
device = tf.device("GPU")

# Get images
data_dir = '/home/alumne/xnap-project-ed_group_05/beta/sport/total/train'
image_size = (256, 256)
Xtrain_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
Xtrain_generator = Xtrain_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=5,
        class_mode=None)

Xtrain = next(Xtrain_generator)
Xtrain = tf.keras.utils.array_utils.to_array(Xtrain, dtype=tf.float32)
Xtrain = Xtrain[..., :1]  # Get only the L channel


# Set up model
model = Sequential()
model.add(InputLayer(input_shape=image_size + (1,)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(UpSampling2D((2, 2)))

optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error)

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 5
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Test images
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)

Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)
#epochs flores 150 y 250 con steps a 33 flores
#deportes

# Move model to GPU
with device:
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="output/first_run")
    history = model.fit_generator(
        image_a_b_gen(batch_size),
        callbacks=[tensorboard],
        epochs=250,
        steps_per_epoch=33,
        validation_data=(Xtest, Ytest)
    )
    model.save_weights("model_weights.h5")
    print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

# Plot loss history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/alumne/xnap-project-ed_group_05/experiments/batch/loss_plot_batch.png')
plt.show()

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Colorization
color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/sport/total/test'):
    color_me.append(img_to_array(load_img('/home/alumne/xnap-project-ed_group_05/beta/sport/total/test/'+filename, target_size=(256, 256))))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))
color_me = tf.convert_to_tensor(color_me, dtype=tf.float32)

output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("/home/alumne/xnap-project-ed_group_05/experiments/batch/l2_mas_steps/img_"+str(i)+".png", lab2rgb(cur))
