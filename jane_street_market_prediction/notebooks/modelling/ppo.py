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


from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.policies import random_tf_policy

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


# -

import pandas as pd
import numpy as np

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

tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(val_py_env)
# -

# ## Hyperparameters

# +
num_iterations = train.shape[0]*2

# networks params
actor_fc_layers=(200, 100)
value_fc_layers=(200, 100)
use_rnns=False

# Replay buffer Collection params
num_environment_steps=25000000,
collect_episodes_per_iteration=1
num_parallel_environments=1
replay_buffer_capacity=5000

# Params for train
num_epochs=1
learning_rate=1e-6

# Params for eval
num_eval_episodes=30
eval_interval=500

# Params for summaries and logging
train_checkpoint_interval=500,
policy_checkpoint_interval=500,
log_interval=50,
summary_interval=50,
summaries_flush_secs=1,
use_tf_functions=True,
debug_summaries=False,
summarize_grads_and_vars=False
initial_collect_steps = 100

# +
global_step = tf.compat.v1.train.get_or_create_global_step()
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

if use_rnns:
    actor_net = (
        actor_distribution_rnn_network
        .ActorDistributionRnnNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=None
      )
    )
    value_net = (
        value_rnn_network
        .ValueRnnNetwork(
            tf_env.observation_spec(),
            input_fc_layer_params=value_fc_layers,
            output_fc_layer_params=None
        )
    )
else:
    actor_net = (
        actor_distribution_network
        .ActorDistributionNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=actor_fc_layers,
            activation_fn=tf.keras.activations.tanh
        )
    )
    value_net = (
        value_network
        .ValueNetwork(
            tf_env.observation_spec(),
            fc_layer_params=value_fc_layers,
            activation_fn=tf.keras.activations.tanh
        )
    )

tf_agent = ppo_clip_agent.PPOClipAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    optimizer,
    actor_net=actor_net,
    value_net=value_net,
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    normalize_observations=False,
    normalize_rewards=False,
    use_gae=True,
    num_epochs=num_epochs,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    train_step_counter=global_step)
tf_agent.initialize()
# -

# ### Replay buffer and initial data collection

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=num_parallel_environments,
    max_length=replay_buffer_capacity
)

# +
environment_steps_metric = tf_metrics.EnvironmentSteps()
step_metrics = [
    tf_metrics.NumberOfEpisodes(),
    environment_steps_metric,
]

train_metrics = step_metrics + [
    tf_metrics.AverageReturnMetric(
        batch_size=num_parallel_environments),
    tf_metrics.AverageEpisodeLengthMetric(
        batch_size=num_parallel_environments),
]
# -

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    tf_env,
    collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_episodes=collect_episodes_per_iteration
)


def train_step():
    trajectories = replay_buffer.gather_all()
    return tf_agent.train(experience=trajectories)


collect_driver.run = common.function(collect_driver.run, autograph=False)
tf_agent.train = common.function(tf_agent.train, autograph=False)
train_step = common.function(train_step)

collect_time = 0
train_time = 0
timed_at_step = global_step.numpy()


def calculate_u_metric(env, policy):
    print("evaluating policy")
  
    time_step = env.reset()
    
    actions = np.array([])
    
    counter = 0
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("start_time", current_time)
    
    while not time_step.is_last():
        action_step = policy.action(time_step)
        actions = np.concatenate((actions, action_step.action.numpy()))
        
        time_step = env.step(action_step.action)
        
        counter += 1
        
        if counter % 10000 ==0 :
            print(counter)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print("cycle_time", current_time)
            
    action_step = policy.action(time_step)
    actions = np.concatenate((actions, action_step.action.numpy()))
            
    eval_df["action"] = pd.Series(data=actions, index = eval_df.index)
    eval_df["trade_reward"] = eval_df["action"]*eval_df["weight"]*eval_df["resp"]
    eval_df["trade_reward_squared"] = eval_df["trade_reward"]*eval_df["trade_reward"]

    tmp = eval_df.groupby(["date"])[["trade_reward", "trade_reward_squared"]].agg("sum")
        
    sum_of_pi = tmp["trade_reward"].sum()
    sum_of_pi_x_pi = tmp["trade_reward_squared"].sum()
    
    print("sum of pi: {sum_of_pi}".format(sum_of_pi = sum_of_pi) )
        
    t = sum_of_pi/np.sqrt(sum_of_pi_x_pi) * np.sqrt(250/tmp.shape[0])
    print("t: {t}".format(t = t) )
    
    u  = np.min([np.max([t, 0]), 6]) * sum_of_pi
    print("u: {u}".format(u = u) )
    
    print("finished evaluating policy")
            
    return t, u


counter = 0
while environment_steps_metric.result() < num_environment_steps:
    print("training step {counter} out of {total_steps} \n".format(counter=counter, total_steps=num_environment_steps))
    global_step_val = global_step.numpy()
    start_time = time.time()
    collect_driver.run()
    collect_time += time.time() - start_time
    
    print("collect time", collect_time)
    
    start_time = time.time()
    total_loss, _ = train_step()
    replay_buffer.clear()
    train_time += time.time() - start_time
    
    print("train time", train_time)
    
    for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=step_metrics)

calculate_u_metric(eval_tf_env, tf_agent.policy)


