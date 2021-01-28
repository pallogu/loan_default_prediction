import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

model = tf.keras.Sequential([
    layers.LSTM(10, input_shape=(1, 4)),
    layers.Dense(2)
])

model.summary()

model(foo.reshape((1, 1, 4)))

foo = np.array([[1, 1.1, 2, 1]])

foo.reshape((1, 1, 4))

model.trainable_variables


