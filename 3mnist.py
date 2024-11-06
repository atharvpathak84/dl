#Build the Image classification model by dividing the model into following 4 stages
#a. Loading and preprocessing the image data
#b. Defining the model’s architecture
#c. Training the model
#d. Estimating the model’s performance

# Importing the required packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt



# a. Loading and preprocessing the image data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()   #Splitting dataset into training and testing
input_shape = (28, 28, 1)

#Make sure values are float so that we can get decimal points after division
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalizing the RGB codes by dividing it to the max RGB value
x_train /= 255
x_test /= 255
print("Shape of training : ", x_train.shape)
print("Shape of testing : ", x_test.shape)



# b. Defining the model's architecture
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.summary()



# c. Training the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2)



# d. Estimating the model's performance
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy=%.3f" %test_acc)
print("Test Loss=%.3f" %test_loss)

#Showing image at position[] from dataset
image = x_train[0]
plt.imshow(np.squeeze(image), cmap='gray')
plt.show()

#Predicting class of image
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
prediction = model.predict([image])
print("Predicted class is: {} ".format(np.argmax(prediction)))