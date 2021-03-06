import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)

# Define and Compile the Neural Network
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

model.compile(optimizer = 'sgd', loss = 'mse')

# y = 2X-1
X = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(X, y, epochs=500)

print(model.predict([10.0]))
