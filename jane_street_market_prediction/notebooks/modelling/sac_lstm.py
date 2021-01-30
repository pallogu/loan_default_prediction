import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy.random import seed
from tensorflow.random import set_seed
import pandas as pd
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories.time_step import TimeStep
import time
import mlflow
import logging
import subprocess
import tensorflow_probability as tfp

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus):
    tf.config.experimental.set_memory_growth(gpus[0], True)

# +
seed(42)
set_seed(42)

tf.keras.backend.set_floatx('float64')
# -

# ## Environments

from environment import MarketEnv

train = pd.read_csv("../etl/train_dataset_after_pca_5fd37b6.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca_5fd37b6.csv")

# +
# eval_df = eval_df[eval_df["date"] < 420]
reward_multiplicator = 100
negative_reward_multiplicator = 100

train_py_env = MarketEnv(
    trades = train,
    features = ["f_{i}".format(i=i) for i in range(40)] + ["feature_0"],
    reward_column = "resp",
    weight_column = "weight",
    include_weight = False,
    discount=0.9,
    reward_multiplicator = reward_multiplicator,
    negative_reward_multiplicator = negative_reward_multiplicator
)

val_py_env = MarketEnv(
    trades = eval_df,
    features = ["f_{i}".format(i=i) for i in range(40)] + ["feature_0"],
    reward_column = "resp",
    weight_column = "weight",
    include_weight = False,
    discount=0.9,
    reward_multiplicator = reward_multiplicator,
    negative_reward_multiplicator = negative_reward_multiplicator
)

tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(val_py_env)
# -

# ## Hyperparameters

# ### General hyperparams

# +
avg_reward_step_size = 1e-2
actor_step_size = 1e-6
critic_step_size = 1e-6
number_of_episodes = 2

tau = 1

# -

# ### Defining Architecture of actor and critic

# +
actor_nn_arch = (
    tf_env.time_step_spec().observation.shape[0],
    128,
    64,
    2
)

actor_dropout = 0.1
critic_dropout = 0.1

critic_nn_arch = (
    tf_env.time_step_spec().observation.shape[0],
    128,
    64,
    1
)


def create_actor_model():
    actor_model = tf.keras.Sequential([
        layers.LSTM(
            actor_nn_arch[1],
            input_shape=(1, actor_nn_arch[0]),
            name="actor_lstm_layer"
        ),
        layers.Dense(
            actor_nn_arch[2],
            activation="relu",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[1], seed=1),
            name="actor_layer_dense_1"
        ),
        layers.Dropout(
            actor_dropout,
            name="actor_dropout_layer"
        ),
        layers.Dense(
            actor_nn_arch[3],
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[2], seed=2),
            name="actor_output_layer"
        )
        
    ])
    
    return actor_model


def create_critic_model():
    critic_model = tf.keras.Sequential([
        layers.LSTM(critic_nn_arch[1], input_shape=(1, critic_nn_arch[0])),
        layers.Dense(
            actor_nn_arch[2],
            activation="relu",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[1], seed=1)
        ),
        layers.Dropout(critic_dropout),
        layers.Dense(
            critic_nn_arch[3],
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[2], seed=2)
        )
        
    ])

    return critic_model


# -

create_actor_model().summary()

create_critic_model().summary()


# ## Definition of metrics

def calculate_u_metric(df, model, boundary=0.0):
    print("evaluating policy")
    with tf.device("/cpu:0"):
        to_predict = df[["f_{i}".format(i=i) for i in range(40)] + ["feature_0"]].values
        
        actions = np.argmax(model(to_predict.reshape((to_predict.shape[0], 1, to_predict.shape[1]))).numpy(), axis=1)
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


# ## AC Agent

class ACAgent():
    def __init__(self, **kwargs):
        self.actor_model = kwargs.get("actor_model")
        self.critic_model = kwargs.get("critic_model")
        self.avg_reward_step_size = tf.constant(
            kwargs.get("avg_reward_step_size"), dtype=np.float64)
        self.avg_reward_step_size_remainer = tf.constant(
            1 - kwargs.get("avg_reward_step_size"), dtype=np.float64)

        actor_step_size = kwargs.get("actor_step_size")
        critic_step_size = kwargs.get("critic_step_size")
        self.tau = tf.constant(kwargs.get("tau", 1), dtype=np.float64)
        self.observation_spec = kwargs.get("observation_spec", tf_env.time_step_spec().observation)

        self.actor_optimizer = keras.optimizers.Adam(
            learning_rate=actor_step_size)
        self.critic_optimizer = keras.optimizers.Adam(
            learning_rate=critic_step_size)

        self.reward = tf.Variable(
            0, dtype=np.float64, name="reward", trainable=False)
        self.delta = tf.Variable(
            0, dtype=np.float64, name="delta", trainable=False)
        self.prev_observation = tf.Variable(tf.zeros_initializer()(
                                            shape=(1, 1, self.observation_spec.shape[0]),
                                            dtype=self.observation_spec.dtype,

                                            ), name=self.observation_spec.name,
                                            trainable=False)
        self.prev_action = tf.Variable(
            0, dtype=np.int32, name="prev_action", trainable=False)

    def init(self, time_step):
        observation = time_step.observation
        reshaped_observation = tf.reshape(observation, shape=(1, 1, self.observation_spec.shape[0]))

        probs = self.policy(reshaped_observation)[0]

        action = tfp.distributions.Bernoulli(
            probs=probs[1], dtype=tf.int32).sample()

        self.prev_observation.assign(reshaped_observation)
        self.prev_action.assign(action)

        return action

    def policy(self, observation):
        return tf.nn.softmax(self.actor_model(observation)/self.tau)

    def train(self, time_step):
        observation = time_step.observation
        reshaped_observation = tf.reshape(observation, shape=(1, 1, self.observation_spec.shape[0]))

        observation_reward = tf.dtypes.cast(tf.reshape(
            time_step.reward, shape=()), tf.float64)

        # self.update_avg_reward(reward)
        # inlined from the function
        self.reward.assign(
            self.avg_reward_step_size_remainer*self.reward +
            self.avg_reward_step_size * observation_reward
        )

        ### self.update_td_error(reward, observation)
        # inlined from the function
        delta = (observation_reward
                 - self.reward
                 + self.critic_model(reshaped_observation)[0][0]
                 - self.critic_model(self.prev_observation)[0][0]
                 )

        self.delta.assign(delta)

        # self.update_critic_model()
        # inlined from the function
        with tf.GradientTape() as tape:
            grad = [-1*self.delta * g for g in tape.gradient(
                self.critic_model(self.prev_observation),
                self.critic_model.trainable_variables
            )]

            self.critic_optimizer.apply_gradients(
                zip(grad, self.critic_model.trainable_variables),
                experimental_aggregate_gradients=False
            )

        # self.update_actor_model()
        # inlined from the function

        prev_action = self.prev_action
        with tf.GradientTape() as tape:

            grad = [-1 * self.delta * g for g in tape.gradient(
                tf.math.log(tf.nn.softmax(self.actor_model(
                    self.prev_observation)/self.tau)[0][self.prev_action]),
                self.actor_model.trainable_variables
            )]

            self.actor_optimizer.apply_gradients(
                zip(grad, self.actor_model.trainable_variables),
                experimental_aggregate_gradients=False
            )

        probs = self.policy(reshaped_observation)[0]

        action = tfp.distributions.Bernoulli(
            probs=probs[1], dtype=tf.int32).sample()

        self.prev_action.assign(action)
        self.prev_observation.assign(reshaped_observation)

        return action


# ## Running of the experiment

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
        
        mlflow.set_tag("agent_type", "sac_lstm")
        mlflow.log_param("actor_nn_layers", actor_nn_arch )
        mlflow.log_param("critic_nn_layers", critic_nn_arch)
        mlflow.log_param("avg_reward_step_size", avg_reward_step_size)
        mlflow.log_param("actor_step_size", actor_step_size)
        mlflow.log_param("critic_step_size", critic_step_size)
        mlflow.log_param("critic_dropout", critic_dropout)
        mlflow.log_param("actor_dropout", actor_dropout)
        mlflow.log_param("tau", tau)
        mlflow.log_param("included_weight", false)
    
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
        subprocess.run(["zip", "-r", "model.zip", "actor_model"])
        mlflow.log_artifact("model.zip")

with tf.device("/cpu:0"):
    run_experiment()
# -

