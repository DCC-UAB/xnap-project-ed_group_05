import tensorflow as tf
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, Concatenate
from keras.utils import array_to_img, img_to_array, load_img
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import os
import wandb
from wandb.keras import WandbCallback

# Set up GPU device
device = tf.device("GPU")

# Get images
X = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train'):
    img = load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_train/' + filename, target_size=(256, 256))
    X.append(img_to_array(img))
X = np.array(X, dtype=float)

# Add channel dimension to the input
X = np.expand_dims(X, axis=-1)

# Start a new wandb run to track this script
config = {
    "learning_rate": 0.001,
    "architecture": "V-Net",
    "dataset": "flors",
    "epochs": 100,
    "regularizador": "no",
    "batch_size": 2,
    "optimizador": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "loss": "mse"
}

wandb.init(
    # Set the wandb project where this run will be logged
    project="my-awesome-project",
    dir="/home/alumne/xnap-project-ed_group_05/experiments/wandb",
    # Track hyperparameters and run metadata
    config=config
)

# Set up train and test data
split = int(0.95 * len(X))
Xtrain = X[:split]
Xtrain = 1.0 / 255 * Xtrain
Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)


def contraction_block(inputs, filters):
    conv1 = Conv3D(filters, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv3D(filters, 3, activation='relu', padding='same')(conv1)
    return conv2


def expansion_block(inputs, skip_conn, filters):
    up = Conv3DTranspose(filters, 2, strides=(2, 2, 2))(inputs)
    concat = Concatenate()([up, skip_conn])
    conv1 = Conv3D(filters, 3, activation='relu', padding='same')(concat)
    conv2 = Conv3D(filters, 3, activation='relu', padding='same')(conv1)
    return conv2


# Add channel dimension to the input shape
input_shape = (256, 256, 1)
inputs = tf.keras.layers.Input(shape=input_shape)
inputs = tf.expand_dims(inputs, axis=-1)

# Contracting Path
conv1 = contraction_block(inputs, 64)
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

conv2 = contraction_block(pool1, 128)
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

conv3 = contraction_block(pool2, 256)
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

conv4 = contraction_block(pool3, 512)

# Expanding Path
up5 = expansion_block(conv4, conv3, 256)
up6 = expansion_block(up5, conv2, 128)
up7 = expansion_block(up6, conv1, 64)
outputs = Conv3D(2, 1, activation='tanh', padding='same')(up7)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Image transformer
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    channel_shift_range=20
)

# Generate training data
batch_size = 10


def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)


# Test images
Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Xtest = tf.convert_to_tensor(Xtest, dtype=tf.float32)

Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
Ytest = Ytest / 128
Ytest = tf.convert_to_tensor(Ytest, dtype=tf.float32)

# Move model to GPU
with device:
    history = model.fit_generator(
        image_a_b_gen(batch_size),
        callbacks=[WandbCallback()],
        epochs=100,
        steps_per_epoch=len(Xtrain) // batch_size,
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
plt.savefig('/home/alumne/xnap-project-ed_group_05/experiments/batch/batch_2/loss_plot_batch_flors.png')
plt.show()

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

# Colorization
color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test'):
    color_me.append(img_to_array(load_img('/home/alumne/xnap-project-ed_group_05/beta/flors/flors_test/' + filename,target_size=(256, 256))))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))
color_me = tf.convert_to_tensor(color_me, dtype=tf.float32)

output = model.predict(color_me)
output = output * 128


# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("/home/alumne/xnap-project-ed_group_05/experiments/batch/batch_2/img_"+str(i)+".png", lab2rgb(cur))