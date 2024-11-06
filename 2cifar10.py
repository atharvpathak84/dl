# a. Import the Necessary Packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random



# b. Load the Training and Testing Data
cifar10 = tf.keras.datasets.cifar10  # Importing the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Splitting into training and testing datasets

# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Check the shape of the input images
#print(x_train.shape)
#print(x_test.shape)



# c. Define the Network Architecture Using Keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),  # Flattening the 32x32x3 images
    keras.layers.Dense(512, activation='relu'),      # Increased number of neurons for complexity
    keras.layers.Dense(10, activation='softmax')      # 10 classes for CIFAR-10
])
model.summary()



# d. Train the Model Using SGD or Adam Optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)  # You can adjust the number of epochs as needed



# e. Evaluate the Network
import numpy as np
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss : %.3f" % test_loss)
print("Accuracy : %.3f" % test_acc)

# Generate a random index to visualize a prediction
n = random.randint(0, len(x_test) - 1)
plt.imshow(x_test[n])
plt.show()
predicted_value = model.predict(x_test)
plt.imshow(x_test[n])
plt.show()
predicted_probabilities = predicted_value[n]

# Get the class with the highest predicted probability
predicted_class = np.argmax(predicted_probabilities)

print("Predicted Class: ", predicted_class)




# f. Plot the Training Loss and Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()