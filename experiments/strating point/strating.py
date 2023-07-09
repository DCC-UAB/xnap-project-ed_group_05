
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
from keras.optimizers import RMSprop, Adam
import random
import tensorflow as tf
import wandb 
from wandb.keras import WandbCallback

# Get images
X = []

for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train'):
    img = load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train/'+filename, target_size=(256, 256))
    X.append(img_to_array(img))
X = np.array(X, dtype=float)

config={
        "learning_rate": "no",
        "architecture": "CNN",
        "dataset": "flors",
        "epochs": 100,
        "regularizador": "no",
        "batch_size": 2, 
        "optimizador": "RMSprop(learning_rate=0.0001)",
        "loss": "mse"
    }
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    dir="/home/alumne/xnap-project-ed_group_05/experiments/wandb",
    # track hyperparameters and run metadata
    config=config
)
# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128

model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer="rmsprop", loss='mse')

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
batch_size=10
# Generate training data
def image_a_b_gen(batch_size):
    while True:
        for batch in datagen.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)


# Train model

model.fit_generator(image_a_b_gen(batch_size), callbacks=[WandbCallback()], epochs=100, steps_per_epoch=len(Xtrain)//batch_size,validation_data=(Xtest, Ytest))

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test'):
    color_me.append(img_to_array(load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test/'+filename, target_size=(256, 256))))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("/home/alumne/xnap-project-ed_group_05/experiments/strating point/img_"+str(i)+".png", lab2rgb(cur))
