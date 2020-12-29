# +
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_dm_control
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS
# -

import pandas as pd

# ### Environments

from environment import MarketEnv

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

# +
eval_df = eval_df[eval_df["date"] < 420]
train_py_env = MarketEnv(
    trades = train,
    features = ["f_{i}".format(i=i) for i in range(40)] + ["weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=0.9
)

val_py_env = MarketEnv(
    trades = eval_df,
    features = ["f_{i}".format(i=i) for i in range(40)] + ["weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=0.9
)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
val_env = tf_py_environment.TFPyEnvironment(val_py_env)


# +
def train_eval(
    num_iterations = 1000000,
    #networks_params
    actor_fc_layers = (400, 300),
    actor_output_fc_layers = (100,),
    actor_lstm_size = (40,),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(300,),
    critic_output_fc_layers=(100,),
    critic_lstm_size=(40,),
    num_parallel_environments=1,
    # Replay buffer Collection params
    inital_collect_episodes=100,
    collect_episodes_per_iteration = 1,
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    critic_learning_rate=3e-4,
    train_sequence_length=20,
    actor_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=0.1,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    critic_learning_rate=3e-4,
    train_sequence_length=20,
    actor_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=0.1,
    gradient_clipping=None,
    use_tf_functions=True,
    # Evaluation params
    num_eval_episodes=30,
    eval_interval=10000,
    
):
    
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes)
        tf_metrics.AverageEpisodeLengthMetric=num_eval_episodes
    ]
