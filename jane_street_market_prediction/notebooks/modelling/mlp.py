# +
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choices


SEED = 1111

np.random.seed(SEED)

train = pd.read_csv('../../input/train.csv')
train = train.query('date > 85').reset_index(drop = True) 
train = train[train['weight'] != 0]

train.fillna(train.mean(),inplace=True)

train['action'] = ((train['resp'].values) > 0).astype(int)


features = [c for c in train.columns if "feature" in c]

f_mean = np.mean(train[features[1:]].values,axis=0)

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X_train = train.loc[:, train.columns.str.contains('feature')]
#y_train = (train.loc[:, 'action'])

y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T

# fit
def create_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)
    
    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )

    return model

epochs = 200
batch_size = 4096
hidden_units = [160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
label_smoothing = 1e-2
learning_rate = 1e-3

tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
clf = create_mlp(
    len(features), 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )

clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
# -

train = pd.read_csv("../etl/train_dataset_with_dates.csv")
eval_df = pd.read_csv("../etl/val_dataset_with_dates.csv")

features = [c for c in train.columns.values if "feature_" in c] + ["date_x", "date_y"]

len(features)

train["label"] = (train["resp"] > 0).astype("int")

eval_df["label"] = (eval_df["resp"] > 0).astype("int")

# +
nn_arch = (
    132,
    512,
    256,
    128,
    64,
    2
)

dropout = 0.1


# -

def create_model():
    model = keras.Sequential([
        layers.Input(shape=nn_arch[0]),
        layers.Dense(
            nn_arch[1],
            activation=tf.nn.swish,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/nn_arch[0], seed=1)
        ),
        layers.Dropout(dropout),
        layers.Dense(
            nn_arch[2],
            activation=tf.nn.swish,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/nn_arch[1], seed=2)
        ),
        layers.Dropout(dropout),
        layers.Dense(
            nn_arch[3],
            activation=tf.nn.swish,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/nn_arch[2], seed=3
            )
        ),
        layers.Dropout(dropout),
        layers.Dense(
            nn_arch[4],
            activation=tf.nn.swish,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/nn_arch[3], seed=3
            )
        ),
        layers.Dropout(dropout),
        layers.Dense(
            nn_arch[5],
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/nn_arch[4], seed=4)
        )
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2),
        optimizer=optimizer,
        metrics=tf.keras.metrics.AUC(name="AUC")
    )


    return model
model = create_model()

model.fit(
    train[features].values,
    train["label"].values.reshape((-1,1)),
    batch_size=256,
    epochs=200,
    verbose=1,
    validation_data=(eval_df[features].values, eval_df["label"].values.reshape((-1,1)))
)

train["label"].values.reshape((-1,1)).shape

train[features].values.shape


def calculate_u_metric(df, model, verbose=0):
    actions = model.predict(df[[c for c in df.columns if "f_" in c] + ["feature_0", "weight"]].values,
                            deterministic=True)[0]
    assert not np.isnan(np.sum(actions))

    sum_of_actions = np.sum(actions)

    df["action"] = pd.Series(data=actions, index=df.index)

    df["trade_reward"] = df["action"] * df["weight"] * df["resp"]

    tmp = df.groupby(["date"])[["trade_reward"]].agg("sum")

    sum_of_pi = tmp["trade_reward"].sum()
    sum_of_pi_x_pi = (tmp["trade_reward"] * tmp["trade_reward"]).sum()
    if sum_of_pi_x_pi == 0:
        return -1000, 0, 0

    t = sum_of_pi / np.sqrt(sum_of_pi_x_pi) * np.sqrt(250 / tmp.shape[0])
    u = np.min([np.max([t, 0]), 6]) * sum_of_pi
    ratio_of_ones = sum_of_actions / len(actions)

    if verbose == 1:
        print("sum of pi: {sum_of_pi}".format(sum_of_pi=sum_of_pi))
        print("t: {t}".format(t=t))
        print("u: {u}".format(u=u))
        print("np_sum(actions)", sum_of_actions)
        print("ration of ones", ratio_of_ones)
        print("length of df", len(actions))

    return t, u, ratio_of_ones


model.predict(eval_df[features].values)

train["label"].values.reshape(1208112, -1)




