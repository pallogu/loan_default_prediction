# # Prediction

import tensorflow as tf
from tf_agents.environments import tf_py_environment
import pickle
from sklearn.decomposition import PCA
from tf_agents.trajectories.time_step import TimeStep
import numpy as np


import sys, os

sys.path.insert(0, "../../input/")
sys.path.insert(0, "../")

from environment import MarketEnv
from etl.ETL import ETL_1, ETL_2

import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set


with open("../etl/etl_1.pkl", "rb") as f:
    etl_1 = pickle.load(f)
    f.close()

with open("../etl/etl_2.pkl", "rb") as f:
    etl_2 = pickle.load(f)
    f.close()

policy = tf.compat.v2.saved_model.load("./model_dqn_128_128_072.policy")

# +
# (test_df, sample_prediction_df) =  next(iter_test)

# +
# trans_1 = test_df.apply(etl_1.fillna_normalize, axis=1)
# trans_2 = trans_1.apply(etl_2.reduce_columns_train, axis=1)

# +
# np.delete(trans_2.values, 0, 1)

# +
# %%time
counter = 0
for (test_df, sample_prediction_df) in iter_test:
    trans_1 = test_df.apply(etl_1.fillna_normalize, axis=1)
    trans_2 = trans_1.apply(etl_2.reduce_columns_train, axis=1)
#     m_py_env = MarketEnv(
#         trades = trans_2,
#         features = ["f_{i}".format(i=i) for i in range(40)] + ["weight"],
#         reward_column = "resp",
#         weight_column = "weight",
#         discount=0.7
#     )
#     m_env = tf_py_environment.TFPyEnvironment(m_py_env)
#     time_step = m_env.reset()

    observation = trans_2.values[1:]
    
    time_step = TimeStep(
        step_type = tf.constant([0], dtype=np.int32),
        reward = tf.constant([0], dtype=np.float32),
        discount = tf.constant([1], dtype=np.float32),
        observation = tf.constant(np.delete(trans_2.values, 0, 1), dtype=np.float64),
    
    )

    action_step = policy.action(time_step)
    sample_prediction_df.action = action_step.action.numpy()[0]
    counter += 1

    env.predict(sample_prediction_df)
    
#     if(counter == 1000):
#         break
# -

env.predictions

counter

env.features

env.predictions.to_csv("predictions.csv")


