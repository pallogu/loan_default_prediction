# # Autoencoder for dimensionality reduction

# ## Importing dependencies 

import tensorflow as tf
import pandas as pd


# ## Importing data

train = pd.read_csv("./train_dataset.csv")
val = pd.read_csv("./val_dataset.csv")

# ## Definition of features

features = [c for c in train.columns.values if "feature" in c][1:]

train_active = train[features]
val_active = val[features]


# ## Definition of auto-encoder

def create_autoencoder(num_target):

    num_features = len(features)


    input_dim = tf.keras.layers.Input(shape = (num_features, ))

    # Encoder Layers
    encoded1 = tf.keras.layers.Dense(
        96,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(input_dim)

    encoded2 = tf.keras.layers.Dense(
        86,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(encoded1)

    encoded3 = tf.keras.layers.Dense(
        64,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(encoded2)


    encoded4 = tf.keras.layers.Dense(
        num_target,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(encoded3)

    # Decoder Layers
    decoded1 = tf.keras.layers.Dense(
        64,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(encoded4)

    decoded2 = tf.keras.layers.Dense(
        86,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(decoded1)

    decoded3 = tf.keras.layers.Dense(
        96,
        activation = 'relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(decoded2)

    decoded4 = tf.keras.layers.Dense(
        num_features,
        activation = None,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01)
    )(decoded3)

    # Combine Encoder and Deocder layers
    autoencoder = tf.keras.Model(inputs = input_dim, outputs = decoded4)
    encoder = tf.keras.Model(inputs = input_dim, outputs = encoded4)
    
    return autoencoder, encoder
    

autoencoder, encoder = create_autoencoder(60)

autoencoder.compile(optimizer = 'adam', loss='mean_squared_error')

autoencoder.fit(
    train_active.values,
    train_active.values,
    epochs= 20,
    batch_size = 128,
    shuffle = False,
    validation_data = (val_active.values, val_active.values)
)


