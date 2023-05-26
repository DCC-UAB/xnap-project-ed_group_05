import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Activation, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
import os
import matplotlib.pyplot as plt

# Get images
X = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05-1/floretes/'):
    img = load_img('/home/alumne/xnap-project-ed_group_05-1/floretes/'+filename, target_size=(256, 256))
    X.append(img_to_array(img))
X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95 * len(X))
Xtrain = X[:split]
Xtrain = 1.0 / 255 * Xtrain

# Model architecture
inputs = Input(shape=(256, 256, 1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)

# Additional layers
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(x)
x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

outputs = UpSampling2D((2, 2))(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='mse')

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model
tensorboard = TensorBoard(log_dir="output/first_run")
history = model.fit(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=50, steps_per_epoch=10)

# Plot loss history
plt.plot(history.history['loss'])
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_plot.png')
plt.show()

# Save model
model.save("model.h5")

# Test images
Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:] / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

# Colorize images
color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05-1/floretes'):
    color_me.append(img_to_array(load_img('/home/alumne/xnap-project-ed_group_05-1/floretes/'+filename, target_size=(256, 256))))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    imsave("/home/alumne/xnap-project-ed_group_05-1/result/img_"+str(i)+".png", lab2rgb(cur))
