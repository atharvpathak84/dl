# Import necessary packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# a. Loading and preprocessing the image data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Splitting dataset
input_shape = (32, 32, 3)

# Reshape and normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print("Shape of training : ", x_train.shape)
print("Shape of testing : ", x_test.shape)

# b. Defining the model's architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

# c. Training the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

# d. Estimating the model's performance
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy=%.3f" % test_acc)
print("Test Loss=%.3f" % test_loss)

# Show an image from CIFAR-10
image = x_test[0]
plt.imshow(image)
plt.show()

# Predict the class of the image
image = np.expand_dims(image, axis=0)  # Reshape to match model input shape
prediction = model.predict(image)
print("Predicted class is: {}".format(np.argmax(prediction)))
