# # Autoencoder for dimensionality reduction

# ## Importing dependencies 

import tensorflow as tf
import pandas as pd
import swifter


# ## Importing data

train = pd.read_csv("./train_dataset.csv")
val = pd.read_csv("./val_dataset.csv")

# ## Definition of features

features = [c for c in train.columns.values if "feature" in c][1:]

rest_of_columns = [c for c in train.columns.values if c not in features]

rest_of_columns

transformed_feature_names = ["feature_{i}".format(i=i) for i in range(1, 36)]

transformed_feature_names



train_active = train[features]
val_active = val[features]

val_active


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


autoencoder, encoder = create_autoencoder(35)

autoencoder.summary()

encoder.summary()



autoencoder.compile(optimizer = 'adam', loss='mean_absolute_error')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
autoencoder.fit(
        train_active.values,
        train_active.values,
        epochs= 100,
        batch_size = 128,
        shuffle = False,
        validation_data = (val_active.values, val_active.values),
        callbacks=[callback]
    )

encoded = encoder.predict(val_active)

encoded.shape

encoder = tf.keras.models.load_model("./encoder_35")


# +
class Encoder():
    def __init__(self, **kwargs):
        self.model = kwargs.get("model")
        self.columns = kwargs.get("columns")
        self.columns_to_keep = kwargs.get("columns_to_keep")
        self.columns_to_keep_train = kwargs.get("columns_to_keep_train")
        self.transformed_columns_names = kwargs.get("transformed_columns_names")
        
    def transform(self, row):
        to_keep = row[self.columns_to_keep]
        to_transform = row[self.columns].values.reshape(1, -1)
        transformed = pd.Series(
            data=self.model.predict(to_transform)[0],
            index=self.transformed_columns_names
        )
        return pd.concat([to_keep, transformed])
    
    def transform_train(self, row):
        to_keep = row[self.columns_to_keep_train]
        to_transform = row[self.columns].values.reshape(1, -1)
        transformed = pd.Series(
            data=self.model.predict(to_transform)[0],
            index=self.transformed_columns_names
        )
        return pd.concat([to_keep, transformed])
        
        
etl_3 = Encoder(
    columns = features,
    columns_to_keep = ["date", "weight", "feature_0"],
    columns_to_keep_train = rest_of_columns,
    transformed_columns_names=transformed_feature_names,
    model = encoder
)
# -

train[:10].apply(etl_3.transform_train, axis=1)

train_after_encoding = train.apply(etl_3.transform_train, axis=1)

val_after_encoding = val.apply(etl_3.transform_train, axis=1)

train_after_enconding.to_csv("./train_dataset_after_encoding.csv", index=False)

val_after_enconding.to_csv("./val_dataset_after_encoding.csv", index=False)

with open("./etl_3.pkl", "wb") as f:
    pickle.dump(etl_3, f)


