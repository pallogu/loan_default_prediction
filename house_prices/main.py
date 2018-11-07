import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib import learn

from sklearn.model_selection import train_test_split

import loadCleanup as dt

train, test = dt.getTrainTestDFs()

y = train.SalePrice
del train['SalePrice']
X = train

dim = len(X.columns.values) + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

X_train = X_train.assign(independent = pd.Series([1] * len(y_train), index=X_train.index))
X_test = X_test.assign(independent = pd.Series([1]*len(y_test), index=X_test.index))

P_train = X_train.as_matrix(columns=None)
P_test = X_test.as_matrix(columns=None)

q_train = np.array(y_train.values).reshape(-1, 1)
q_test = np.array(y_test.values).reshape(-1, 1)

P = tf.placeholder(tf.float32, [None, dim])
q = tf.placeholder(tf.float32, [None, 1])
T = tf.Variable(tf.ones([dim, 1]))

bias = tf.Variable

model = tf.estimator.DNNRegressor(hidden_units=[20, 20], model_dir='./dnnr')

model.train(P_train, q_train)
if __name__ == "main":
    tf.logging.set.verbosity(tf.logging.INFO)
    tf.app.run(main=main)