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


# -

# ## Hyperparameters

# ### General hyperparams

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
    # Params for summaries and logging
    train_checkpoint_interval=10000
    policy_checkpoint_interval=5000
    rb_checkpoint_interval=50000
    log_interval=1000
    summary_interval=1000
    summaries_flush_secs=10
    debug_summaries=False
    summarize_grads_and_vars=False
    eval_metrics_callback=None
    root_dir = "./"
    
    eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

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
    train_metrics = [
        tf_metrics.NumberOfEpisodes(prefix='Train'),
        env_steps,
        average_return,
        tf_metrics.AverageEpisodeLengthMetric(
            prefix='Train',
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size),
    ]
    
    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy
    
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'train'),
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
    
    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_episodes=initial_collect_episodes)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
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

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=env_steps.result(),
        summary_writer=summary_writer,
        summary_prefix='Eval',
    )
    if eval_metrics_callback is not None:
        eval_metrics_callback(results, env_steps.result())
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)


with tf.device("/cpu:0"):
    train()























def calculate_u_metric(df, model, boundary=0.0):
    print("evaluating policy")
    with tf.device("/cpu:0"):
        actions = np.argmax(model(df[["f_{i}".format(i=i) for i in range(40)] + ["weight"]].values).numpy(), axis=1)
        assert not np.isnan(np.sum(actions))

    #     probs = tf.nn.softmax(model(df[["f_{i}".format(i=i) for i in range(40)] + ["weight"]].values)).numpy()
    #     probs_df = pd.DataFrame(probs, columns=["0", "1"])
    #     probs_df["probs_diff"] = probs_df["1"] - probs_df["0"]
    #     probs_df["action"] = probs_df["probs_diff"] > boundary
    #     probs_df["action"] = probs_df["action"].astype("int")

        sum_of_actions = np.sum(actions)
        print("np_sum(actions)", sum_of_actions)

    #     df["action"] = probs_df["action"]
        df["action"] = pd.Series(data=actions, index=df.index)
        df["trade_reward"] = df["action"]*df["weight"]*df["resp"]
        df["trade_reward_squared"] = df["trade_reward"]*df["trade_reward"]

        tmp = df.groupby(["date"])[["trade_reward", "trade_reward_squared"]].agg("sum")

        sum_of_pi = tmp["trade_reward"].sum()
        sum_of_pi_x_pi = tmp["trade_reward_squared"].sum()

        print("sum of pi: {sum_of_pi}".format(sum_of_pi = sum_of_pi) )

        if sum_of_pi_x_pi == 0.0:
            return 0, 0, 0

        t = sum_of_pi/np.sqrt(sum_of_pi_x_pi) * np.sqrt(250/tmp.shape[0])
        # print("t: {t}".format(t = t) )

        u = np.min([np.max([t, 0]), 6]) * sum_of_pi
        print("u: {u}".format(u = u) )

        ratio_of_ones = sum_of_actions/len(actions)
        
        return t, u, ratio_of_ones


# +
# %%time

agent = ACAgent(
    actor_model=create_actor_model(),
    critic_model=create_critic_model(),
    avg_reward_step_size=avg_reward_step_size,
    actor_step_size=actor_step_size,
    critic_step_size=critic_step_size,
    tau = tau
)

agent.train = tf.function(agent.train)
agent.init = tf.function(agent.init)

def run_experiment():
    with mlflow.start_run():
        
        mlflow.set_tag("agent_type", "sac")
        mlflow.log_param("actor_nn_layers", actor_nn_arch )
        mlflow.log_param("critic_nn_layers", critic_nn_arch)
        mlflow.log_param("avg_reward_step_size", avg_reward_step_size)
        mlflow.log_param("actor_step_size", actor_step_size)
        mlflow.log_param("critic_step_size", critic_step_size)
        mlflow.log_param("critic_dropout", critic_dropout)
        mlflow.log_param("actor_dropout", actor_dropout)
        mlflow.log_param("tau", tau)
        mlflow.log_param("reward_multiplicator", reward_multiplicator)
        mlflow.log_param("negative_reward_multiplicator", negative_reward_multiplicator)
    
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        for epoch in range(number_of_episodes):
            time_step = tf_env.reset()
            action = agent.init(time_step)
            counter = 0
            for _ in range(train.shape[0]):
                time_step = tf_env.step(action)
                action = agent.train(time_step)
                counter += 1

                if counter % 100000 == 0:
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(epoch, counter, current_time)
                    t_eval, u_eval, ratio_of_ones_eval = calculate_u_metric(eval_df, agent.actor_model)
                    t_train, u_train, ratio_of_ones_train = calculate_u_metric(train, agent.actor_model)
                    mlflow.log_metrics({
                        "t_eval": t_eval,
                        "u_eval": u_eval,
                        "t_train": t_train,
                        "u_train": u_train,
                        "ratio_of_ones_eval": ratio_of_ones_eval,
                        "ratio_of_ones_train": ratio_of_ones_train
                    })
            agent.actor_model.save("./actor_model")         
            subprocess.run(["zip", "-r", "model_{epoch}.zip".format(epoch=epoch), "actor_model"])
            mlflow.log_artifact("model_{epoch}.zip".format(epoch=epoch))


run_experiment()
# -
# ### Debugging

# +
# agent = ACAgent(
#     actor_model=create_actor_model(),
#     critic_model=create_critic_model(),
#     avg_reward_step_size=avg_reward_step_size,
#     actor_step_size=actor_step_size,
#     critic_step_size=critic_step_size,
#     verbose=True
# )
# time_step = tf_env.reset()
# action = agent.init(time_step) 
# print(action)

# time_step = tf_env.step(action)
# action = agent.train(time_step)
# action
# -

agent.actor_model.summary()



actor_model_test = keras.Sequential([
    layers.Input(shape=4),
    layers.Dense(
        4,
        activation="relu",
        kernel_initializer=tf.keras.initializers.Constant(value=1)
    ),
    layers.Dense(
        2,
        kernel_initializer=tf.keras.initializers.Constant(value=1)
    )
])

test_observation = tf.constant(np.array([[1, 1, 1, 1]]), dtype=np.float64)

actor_model_test(test_observation)

# +
with tf.GradientTape() as tape:
    grad = tape.gradient(
        tf.math.log(tf.nn.softmax(actor_model_test(test_observation))[0][1]),
        actor_model_test.trainable_variables
    )
    
grad
# -

actor_model_test(test_observation)[0][1]

agent.actor_model.save("model_2")

calculate_u_metric(train, agent.actor_model)

calculate_u_metric(train, agent.actor_model)

calculate_u_metric(train, agent.actor_model, boundary=0.9)

probs = tf.nn.softmax(agent.actor_model(train[["f_{i}".format(i=i) for i in range(40)] + ["weight"]].values)).numpy()

probs_ds = pd.DataFrame(data=probs, columns=["0", "1"])

probs_ds["prob_diff"] = probs_ds["1"] - probs_ds["0"]

probs_ds["action"] = probs_ds["prob_diff"] > 0.8

probs_ds["action"] = probs_ds["action"].astype("int")

probs_ds["action"].sum()

tf.constant(3, dtype=np.float32)

b = tf.Variable(8, dtype=np.float32, name="b", trainable=False)

b.assign(9)

b

np.int

timestep= train_py_env.reset()

timestep.reward


