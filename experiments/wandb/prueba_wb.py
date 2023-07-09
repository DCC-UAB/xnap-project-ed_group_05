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
X = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train'):
    img = load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train/'+filename, target_size=(256, 256))
    X.append(img_to_array(img))
X = np.array(X, dtype=float)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    dir="/home/alumne/xnap-project-ed_group_05/experiments/wandb",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "flors",
    "epochs": 250,
    "regularizador": "L2",
    "batch_size": 5, 
    "steps": 34,
    "optimizador": "RMSprop",
    "loss": "MSE"

    }
)

#config = wandb.config()

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain
Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
#l2
'''
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
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

'''
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

optimizer = RMSprop(learning_rate = 0.001)
model.compile(optimizer='rmsprop', loss='mse')


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
        #print(f'min {X_batch.min()} max {X_batch.max()}  {X_batch.shape()}')
        #print(X_batch)
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
samples=166
# Move model to GPU
with device:
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="output/first_run")
    history = model.fit_generator(
        image_a_b_gen(batch_size),
        callbacks=[tensorboard],
        epochs=100,
        steps_per_epoch=samples//batch_size,
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
plt.savefig('/home/alumne/xnap-project-ed_group_05/experiments/wandb/relacio_bona/loss_wb.png')
plt.show()

wandb.log({"Training Loss": history.history['loss'][-1]})
wandb.log({"Validation Loss": history.history['val_loss'][-1]})


# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#show model
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True)

# Colorization
color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test'):
    color_me.append(img_to_array(load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test/'+filename, target_size=(256, 256))))
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
    imsave("/home/alumne/xnap-project-ed_group_05/experiments/wandb/relacio_bona/img_"+str(i)+".png", lab2rgb(cur))
