from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

# Función rgb2gray modificada para TensorFlow
import tensorflow.keras.backend as K

def rgb2gray(rgb):
    # Asegurarse de que el tensor sea de tipo float32 y tenga rango [0, 1]
    rgb = K.cast(rgb, tf.float32) / 255.0
    # Calcular la conversión a escala de grises
    gray = 0.2989 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Ajustar la forma del tensor de salida
    gray = K.expand_dims(gray, axis=-1)
    return gray

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

# Define the model
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

# Load pre-trained VGG16 model (excluding the top layers)
vgg_model = VGG16(weights='imagenet', include_top=False)

# Define the perceptual loss function
def perceptual_loss(y_true, y_pred):
    # Compute MSE loss
    mse_loss = tf.losses.mean_squared_error(y_true, y_pred)

    # Compute perceptual loss using VGG16 features
    y_true_gray = rgb2gray(y_true)
    y_pred_gray = rgb2gray(y_pred)
    y_true_features = vgg_model(y_true_gray)
    y_pred_features = vgg_model(y_pred_gray)
    perceptual_loss = tf.losses.mean_squared_error(y_true_features, y_pred_features)

    # Combine the losses (adjust the weights as needed)
    total_loss = mse_loss + 0.1 * perceptual_loss

    return total_loss

# Compile the model with the custom loss function
model.compile(optimizer='rmsprop', loss='mse')

# Image transformer
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)


batch_size = 10
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[..., 0]
        Y_batch = lab_batch[..., 1:] / 128
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)

# Train the model
tensorboard = TensorBoard(log_dir="output/first_run")
history = model.fit(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=50, steps_per_epoch=10)


# Plot loss history
plt.plot(history.history['loss'])
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_plot.png')
plt.show()

# Save the model
model.save('model.h5')

# Test images
Xtest = rgb2lab(1.0 / 255 * X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

# Colorize images
color_me = []
for filename in os.listdir('/home/alumne/xnap-project-ed_group_05-1/floretes'):
    color_me.append(img_to_array(load_img('/home/alumne/xnap-project-ed_group_05-1/floretes/'+filename, target_size=(256, 256))))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255 * color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape + (1,))

# Generate colorized images
output = model.predict(color_me)
output = output * 128

# Save colorized images
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("/home/alumne/xnap-project-ed_group_05-1/result/img_"+str(i)+".png", lab2rgb(cur))
