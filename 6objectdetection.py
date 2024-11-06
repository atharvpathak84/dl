from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'dataset/mnist-jpg/mnist-jpg/train/'
test_dir = 'dataset/mnist-jpg/mnist-jpg/test/'


img_gen = ImageDataGenerator(rescale=1.0/255)
data_gen = img_gen.flow_from_directory(
    train_dir,
    target_size=(32,32),
    batch_size=5000,
    shuffle=True,
    class_mode='categorical'
)


x_train, y_train = data_gen[0]
x_test, y_test = data_gen[2]



from tensorflow.keras.applications import VGG16
path = 'dataset/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

vgg_model = VGG16(weights=path,include_top=False, input_shape=(32,32,3))



for layer in vgg_model.layers:
    layer.trainabler=False
    
    

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout



custom_classifier = keras.Sequential([
    Flatten(input_shape=(1,1,512)),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
    
])

model = keras.Sequential([
    vgg_model,
    custom_classifier
])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=1, validation_data=(x_test,y_test))




for layer in vgg_model.layers[:-4]:
    layer.trainable = True
    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1000, epochs=1, validation_data=(x_test,y_test))




loss, acc = model.evaluate(x_test, y_test)
print(loss, " ", acc)


pred = model.predict(x_test)


labels = list(data_gen.class_indices.keys())



import matplotlib.pyplot as plt
import numpy as np
plt.imshow(x_test[10])
plt.title(str(labels[np.argmax(pred[10])]))
print(str(labels[np.argmax(y_test[10])]))



y_test[10]
