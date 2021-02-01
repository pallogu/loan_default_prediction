# +
from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import pandas as pd

import mlflow

import time

# -

from tf_agents.policies.policy_saver import PolicySaver

# ### Environment
#

from environment import MarketEnv

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

eval_df = eval_df[eval_df["date"] < 420]

eval_df.shape

# +
discount = 0.75

train_py_env = MarketEnv(
    trades = train,
    features = ["f_{i}".format(i=i) for i in range(40)] + ["weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=discount
)

val_py_env = MarketEnv(
    trades = eval_df,
    features = ["f_{i}".format(i=i) for i in range(40)] + ["weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=discount
)
# -

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
val_env = tf_py_environment.TFPyEnvironment(val_py_env)

# ### Hyperparameters

# +
num_iterations = train.shape[0]

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = num_iterations*2

batch_size = 128
learning_rate = 1e-6
log_interval = np.floor(num_iterations / 1000)

num_eval_episodes = 10
eval_interval = np.floor(num_iterations / 50)
# -

# ### Agent

# +
fc_layer_params = (32, 32,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# +
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()
# -

# ### Replay buffer and initial data collection

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = replay_buffer_max_length
)

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


def collect_data(env, policy, buffer, steps):
    for i in range(steps):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        buffer.add_batch(
            trajectory.from_transition(
                time_step,
                action_step,
                next_time_step
            )
        )


collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls = -1,
    sample_batch_size = batch_size,
    num_steps = 2
).prefetch(3)

iterator = iter(dataset)


# ### Metrics and Evaluation

def calculate_u_metric(env, policy):
    print("evaluating policy")
  
    time_step = env.reset()
    
    actions = np.array([])
    
    counter = 0
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("start_time", current_time)
    
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        actions = np.concatenate((actions, action_step.action.numpy()))
        
        time_step = env.step(action_step.action)
        
        counter += 1
        
        if counter % 10000 ==0 :
            print(counter)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print("cycle_time", current_time)
            
    action_step = agent.policy.action(time_step)
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


# ### Training the agent

def run_experiment():
    with mlflow.start_run():
#         num_iterations = 1000
        mlflow.set_tag("agent_type", "dqn")
        mlflow.log_param("num_act_units", fc_layer_params)
        mlflow.log_param("num_iterations", num_iterations)
        mlflow.log_param("initial_collect_steps", initial_collect_steps)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.set_tag("data_set", "initial_dataset_after_pca")
        mlflow.log_param("discount", discount)
        mlflow.log_param("run", 2)
        
        agent.train = common.function(agent.train)
        
        agent.train_step_counter.assign(0)
        
        
        for _ in range(num_iterations):
            collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
            
            experience, unused_info = next(iterator)
            
            train_loss = agent.train(experience).loss
            
            step = agent.train_step_counter.numpy()
                        
            if (step - 1) % log_interval == 0:
                print("step: ", step)
                mlflow.log_metric("loss", train_loss.numpy())
                
            if _ % eval_interval == 0:
                t, u = calculate_u_metric(val_env, agent.policy)
                mlflow.log_metric("u_metric", u)
                mlflow.log_metric("t_metric", t)
                mlflow.log_metric("step", step)

# %%time
run_experiment()

saver = PolicySaver(agent.policy, batch_size=None)

saver.save("model_dqn_32_32_075_short_2.policy")



calculate_u_metric(val_env, agent.policy)


