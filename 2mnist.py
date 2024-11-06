#Implementing Feedforward neural networks with Keras and TensorFlow for classification of hand-written MNIST dataset using below steps:
#a. Import the necessary packages
#b. Load the training and testing data
#c. Define the network architecture using Keras
#d. Train the model using SGD with 11 epochs
#e. Evaluate the network
#f. Plot the training loss and accuracy

# a. Importing the necessary packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random



# b. Load the training and testing data
mnist = tf.keras.datasets.mnist      #importing the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()    #splitting it into training and testing dataset

x_train = x_train / 255
x_test = x_test / 255

# input image ki size kaise pata lagi
#print(x_train.shape)
#print(x_test.shape)



# c. Define the network architecture with keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()



# d. Train the model using SGD
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=11)



# Evaluate the network
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss : %.3f" %test_loss)
print("Accuracy : %.3f" %test_acc)

#random number generate kiya
n = random.randint(0, 9999)
plt.imshow(x_test[n])
plt.show()
predicted_value = model.predict(x_test)
plt.imshow(x_test[n])
plt.show()
print("Predicted Value : ",predicted_value[n])



# f. Plot the training loss & accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()