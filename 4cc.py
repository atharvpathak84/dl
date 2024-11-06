#Implement anomaly detection for given credit card dataset using Autoencoder and build the model by using the following steps:
#a. Import required libraries
#b. Upload / access the dataset
#c. Encoder converts it into latent representation
#d. Decoder networks convert it back to the original input
#e. Compile the models with Optimizer, Loss, and Evaluation Metrics

import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')
df

df = df.drop(['Time', 'Class'], axis = 1)
df


# Preprocess the data
x_train, x_test = train_test_split(df, test_size=0.2)
print(x_train.shape[1])
print(x_test.shape[1])


#Encoder: Compresses data into a latent representation
#The encoder reduces the input dimensionality. The last layer, with 20 neurons, represents the latent space.
encoder = tf.keras.models.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(20, activation='relu')
])
#Decoder: Reconstructs the data from the latent representation.
#The decoder reconstructs the input to its original dimensions. The last layer uses a linear activation, suitable for reconstructing continuous input features.
decoder = tf.keras.models.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(x_train.shape[1], activation='linear')  # Use linear activation for reconstruction
])
#Autoencoder Model : The encoder and decoder are combined in a single model.
model = tf.keras.models.Sequential([
    encoder,
    decoder
])


#The model is compiled with the adam optimizer and a mean_squared_error loss, as we are reconstructing the input data.
#Training occurs on normal transactions from x_train, with validation on x_test. The model learns to reconstruct typical transactions.
model.compile(optimizer='adam', loss ='mean_squared_error')
history = model.fit(
    x_train,
    x_train,
    validation_data=(x_test,x_test),
    epochs=5,
    batch_size = 100,
    shuffle=True
)


#Plot Training Loss
import seaborn as sns
sns.lineplot(model.history.history)


#Predictions are made on x_test.
#The Mean Squared Error (MSE) between original and reconstructed data is computed. Higher MSE indicates that the input is less similar to normal transactions.
#A threshold is set at the 95th percentile of MSE values, above which samples are considered anomalies.
predictions = model.predict(x_test)
mse = np.mean(np.power(x_test - predictions, 2), axis=1)


#7. Identify and Count Anomalies
#The anomalies are identified as transactions with an MSE higher than the threshold.
#The total number of anomalies is printed.
threshold = np.percentile(mse, 95)  # Adjust the percentile as needed
threshold


anomalies = mse > threshold
#Plot the anomalies
import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
plt.plot(mse, marker='o', linestyle='', markersize=3, label='MSE', )
plt.axhline(threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.xlabel('Sample Index')
plt.ylabel('MSE')
plt.title('Anomaly Detection Results')
plt.legend()
plt.show()