import tensorflow as tf
from keras.optimizers import RMSprop, Adam
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

#### salen las imagenes rojizas enteras y la val loss es ctt, con adam a 0.0001 salen las imagenes en blanco y negro

# Set up GPU device
device = tf.device("GPU")

# Get images
X = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train'):
    img = load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train/'+filename, target_size=(256, 256))
    X.append(img_to_array(img))
X = np.array(X, dtype=float)

# start a new wandb run to track this script
config={
        "learning_rate": 0.0001,
        "architecture": "CNN",
        "dataset": "flors",
        "epochs": 250,
        "regularizador": "l2",
        "batch_size": 7, 
        "optimizador": RMSprop(learning_rate = 0.0001),
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
split = int(0.95 * len(X))
Xtrain = X[:split]

Xtrain_lab = rgb2lab(Xtrain)

# Normalize training data
Xtrain = Xtrain / 255.0

# Load and normalize test data
Xtest = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test'):
    img = load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test/'+filename, target_size=(256, 256))
    Xtest.append(img_to_array(img))
Xtest = np.array(Xtest, dtype=float)
Xtest_lab = rgb2lab(Xtest)
Xtest = Xtest / 255.0  # Normalizar en el rango [0, 1]
Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)

# Convert training images to Lab color space
Xtrain_lab = Xtrain_lab[:, :, :, 0]  # Extract L channel
Xtrain_lab = Xtrain_lab.reshape(Xtrain_lab.shape + (1,))

# Convert training labels to Lab color space and normalize
Ytrain_lab = Xtrain
Ytrain_lab = Ytrain_lab[:, :, :, 1:]  # no 128

# Convert test images to Lab color space
Xtest_lab = Xtest_lab[:, :, :, 0]  # Extract L channel
Xtest_lab = Xtest_lab.reshape(Xtest_lab.shape + (1,))

# Convert test labels to Lab color space and normalize
Ytest_lab = Xtest
Ytest_lab = Ytest_lab[:, :, :, 1:] 
# Shuffle the training data

indices = np.arange(len(Xtrain_lab))

np.random.shuffle(indices)

indices_tensor = tf.constant(indices)  # Convertir a tensor

Xtrain_lab_shuffled = tf.gather(Xtrain_lab, indices_tensor)

Ytrain_lab_shuffled = tf.gather(Ytrain_lab, indices_tensor)
# Create the model
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

# Compile the model
model.compile(optimizer=config["optimizador"], loss=config["loss"])

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        channel_shift_range=20)

# Generate training data
def image_a_b_gen (batch_size):
    for batch in datagen.flow(Xtrain_lab, Ytrain_lab, batch_size=config["batch_size"]):
        yield (batch[0], batch[1])

# Move model to GPU
with device:
    history = model.fit_generator(
        image_a_b_gen(config["batch_size"]),
        callbacks=[WandbCallback()],
        epochs=config["epochs"],
        steps_per_epoch=len(Xtrain) // config["batch_size"],
        validation_data=(Xtest_lab, Ytest_lab)
    )

# Plot loss history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/alumne/xnap-project-ed_group_05/experiments/wandb/shuffle+ruido/loss_wb.png')
plt.show()

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
'''
#show model
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True
)
'''
# Leer y preprocesar las imágenes de entrada
color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test'):
    img = img_to_array(load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test/' + filename, target_size=(256, 256)))
    color_me.append(img)

# Convertir a array de tipo float y normalizar
color_me = np.array(color_me, dtype=float)
color_me = color_me / 255.0  # Normalizar en el rango [0, 1]

# Convertir a Lab color space
color_me_lab = rgb2lab(color_me)

# Extraer el canal L
color_me_l = color_me_lab[:, :, :, 0]
color_me_l = color_me_l.reshape(color_me_l.shape + (1,))
color_me_l = color_me_l / 100.0  # Normalizar en el rango [0, 1] dividiendo por 100.0
color_me_l = tf.convert_to_tensor(color_me_l, dtype=tf.float32)

# Obtener la salida del modelo
output = model.predict(color_me_l)

# Deshacer la normalización
output = output * 128.0  # Multiplicar por 128.0 en lugar de 100.0

# Generar imágenes de salida en color
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me_l[i][:, :, 0] * 100.0  # Multiplicar por 100.0 para deshacer la normalización
    cur[:, :, 1:] = output[i]
    cur = lab2rgb(cur)
    cur = np.clip(cur, 0, 1)  
    cur = np.uint8(cur * 255.0) 
    # # Asegurarse de que los valores estén en el rango [0, 1]
    imsave("/home/alumne/xnap-project-ed_group_05/experiments/wandb/shuffle+ruido/img_" + str(i)+".png", cur)
