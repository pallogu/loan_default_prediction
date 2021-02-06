# %load_ext autoreload
# %autoreload 2

# +
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time


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


# +
# tf.keras.backend.set_floatx('float64')
# -

# ## Environments

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

tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(val_py_env)

def train():
    num_iterations=1000000
    # Params for networks.
    actor_fc_layers=(128, 64)
    actor_output_fc_layers=(64,)
    actor_lstm_size=(32,)
    critic_obs_fc_layers=None
    critic_action_fc_layers=None
    critic_joint_fc_layers=(128,)
    critic_output_fc_layers=(64,)
    critic_lstm_size=(32,)
    num_parallel_environments=1
    # Params for collect
    initial_collect_episodes=1
    collect_episodes_per_iteration=1
    replay_buffer_capacity=1000000
    # Params for target update
    target_update_tau=0.05
    target_update_period=5
    # Params for train
    train_steps_per_iteration=1
    batch_size=256
    critic_learning_rate=3e-4
    train_sequence_length=20
    actor_learning_rate=3e-4
    alpha_learning_rate=3e-4
    td_errors_loss_fn=tf.math.squared_difference
    gamma=0.99
    reward_scale_factor=0.1
    gradient_clipping=None
    use_tf_functions=True
    # Params for eval
    num_eval_episodes=30
    eval_interval=10000

    log_interval=1000
    summaries_flush_secs=10
    debug_summaries=False
    summarize_grads_and_vars=False
    root_dir = "./"
    
    summary_writer = tf.compat.v2.summary.create_file_writer(
    root_dir, flush_millis=summaries_flush_secs * 1000)
    summary_writer.set_as_default()


    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()
    
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=actor_fc_layers,
        lstm_size=actor_lstm_size,
        output_fc_layer_params=actor_output_fc_layers,
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    critic_net = critic_rnn_network.CriticRnnNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        lstm_size=critic_lstm_size,
        output_fc_layer_params=critic_output_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
    
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
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]
    
    env_steps = tf_metrics.EnvironmentSteps(prefix='Train')
    average_return = tf_metrics.AverageReturnMetric(
        prefix='Train',
        buffer_size=num_eval_episodes,
        batch_size=tf_env.batch_size)

    
    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy
    
    
    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer,
        num_episodes=initial_collect_episodes)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=replay_observer,
        num_episodes=collect_episodes_per_iteration)

    if use_tf_functions:
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)
        
    if env_steps.result() == 0 or replay_buffer.num_frames() == 0:
        logging.info(
          'Initializing replay buffer by collecting experience for %d episodes '
          'with a random policy.', initial_collect_episodes)
        initial_collect_driver.run()


    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    
    time_acc = 0
    env_steps_before = env_steps.result().numpy()

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
      # Reduce filter_fn over full trajectory sampled. The sequence is kept only
      # if all elements except for the last one pass the filter. This is to
      # allow training on terminal steps.
      return tf.reduce_all(~trajectories.is_boundary()[:-1])
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=train_sequence_length+1).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(5)
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    if use_tf_functions:
        train_step = common.function(train_step)

    for _ in range(num_iterations):
        start_env_steps = env_steps.result()
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )
        episode_steps = env_steps.result() - start_env_steps
        # TODO(b/152648849)
        for _ in range(episode_steps):
            for _ in range(train_steps_per_iteration):
                train_step()





with tf.device("/cpu:0"):
    train()



