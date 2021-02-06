# %load_ext autoreload
# %autoreload 2

# +
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import reverb


from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import critic_rnn_network, critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_dm_control
from tf_agents.environments import tf_py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network, actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils


import pandas as pd

import time
import mlflow
import logging
import subprocess
import tensorflow_probability as tfp
# -

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus):
    tf.config.experimental.set_memory_growth(gpus[0], True)

from environment import MarketEnvContinuous

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

# +
# eval_df = eval_df[eval_df["date"] < 420]
reward_multiplicator = 100
negative_reward_multiplicator = 103.91
discount = 0.1

train_py_env = MarketEnvContinuous(
    trades = train,
    features = [c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=discount,
    reward_multiplicator = reward_multiplicator,
    negative_reward_multiplicator = negative_reward_multiplicator
)

val_py_env = MarketEnvContinuous(
    trades = eval_df,
    features = [c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=discount,
    reward_multiplicator = 1,
    negative_reward_multiplicator = 1
)

collect_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(val_py_env)
# -

# ## Hyperparameters

# +
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}


# +
time_step_spec = tf_env.time_step_spec()
observation_spec = time_step_spec.observation
action_spec = tf_env.action_spec()

critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

# -

actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))


# +
def train_step():
    experience, _ = next(iterator)
    return tf_agent.train(experience)

tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

tf_agent.initialize()

# -

# ## Replay Buffer

rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)



