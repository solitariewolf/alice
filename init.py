import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Load X and Y from .npy files
X = np.load('DATA/X.npy')
y = np.load('DATA/y.npy')

# Define a simple neural network model
model = Sequential()
print("Shape of X:", X.shape)
print("First element of X:", X[0])
model.add(Dense(512, input_shape=(X.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(set(y)), activation='softmax'))

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.05)

# Save the trained model
model.save('DATA/model.h5')



