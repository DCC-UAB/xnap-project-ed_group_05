from keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
# Load and preprocess the image
image = img_to_array(load_img('/home/alumne/xnap-project-ed_group_05-1/alpha/images/niñosplaya.jpg'))
image = np.array(image, dtype=float)
X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:] / 128

# Resize the images
X = resize(X, (400, 400))
Y = resize(Y, (400, 400, 2))
X = X.reshape(1, 400, 400, 1)
Y = Y.reshape(1, 400, 400, 2)

# Build the neural network
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

# Compile the model
optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Train the model
history = model.fit(x=X, y=Y, batch_size=1, epochs=1000)

# Evaluate the model
print(model.evaluate(X, Y, batch_size=1))

# Generate output
output = model.predict(X)
output *= 128

# Output colorizations
cur = np.zeros((400, 400, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("/home/alumne/xnap-project-ed_group_05-1/alpha/results/img_result.png", lab2rgb(cur))
imsave("/home/alumne/xnap-project-ed_group_05-1/alpha/results/img_gray_version.png", rgb2gray(lab2rgb(cur)))

# Plot loss
loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_capa.png')
plt.show()

