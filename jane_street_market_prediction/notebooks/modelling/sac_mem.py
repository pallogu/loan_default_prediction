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

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

train.head()

reward_multiplicator = 100
negative_reward_multiplicator = 103.9

# +
train_py_env = MarketEnv(
    trades = train,
    features = [c for c in train.columns if "f_" in c] + ["feature_0", "weight"],
    reward_column = "resp",
    weight_column = "weight",
    discount=0.9,
    include_weight=True,
    reward_multiplicator = reward_multiplicator,
    negative_reward_multiplicator = negative_reward_multiplicator
)

tf_env = tf_py_environment.TFPyEnvironment(train_py_env)

# +
val_py_env = MarketEnv(
    trades = eval_df,
    features = [c for c in train.columns if "f_" in c] + ["feature_0", "weight"],
    reward_column = "resp",
    weight_column = "weight",
    include_weight=True,
    discount=0.9,
    reward_multiplicator = reward_multiplicator,
    negative_reward_multiplicator = negative_reward_multiplicator
)

eval_tf_env = tf_py_environment.TFPyEnvironment(val_py_env)
# -

# ## Hyperparameters

# ### General hyperparams

# +
avg_reward_step_size = 0.95
actor_step_size = 1e-6
critic_step_size = 1e-6
number_of_episodes = 4
leaky_relu_alpha = 0.01

tau = 1

# -

# ### Defining Architecture of actor and critic

# +
leaky_relu = tf.keras.layers.LeakyReLU(alpha = leaky_relu_alpha)

actor_nn_arch = (
    tf_env.time_step_spec().observation.shape[0],
    tf_env.time_step_spec().observation.shape[0]*3,
    tf_env.time_step_spec().observation.shape[0]*7,
    tf_env.time_step_spec().observation.shape[0]*7,
    tf_env.time_step_spec().observation.shape[0]*3,
    2
)

actor_dropout = 0.15
critic_dropout = 0.15

critic_nn_arch = (
    tf_env.time_step_spec().observation.shape[0],
    tf_env.time_step_spec().observation.shape[0]*3,
    tf_env.time_step_spec().observation.shape[0]*7,
    tf_env.time_step_spec().observation.shape[0]*7,
    tf_env.time_step_spec().observation.shape[0]*3,
    1
)


def create_actor_model():
    actor_model = keras.Sequential([
        layers.Input(shape=actor_nn_arch[0]),
        layers.Dense(
            actor_nn_arch[1],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[0], seed=1)
        ),
        layers.Dropout(actor_dropout),
        layers.Dense(
            actor_nn_arch[2],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[1], seed=2)
        ),
        layers.Dropout(actor_dropout),
        layers.Dense(
            actor_nn_arch[3],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[2], seed=3
            )
        ),
        layers.Dropout(actor_dropout),
        layers.Dense(
            actor_nn_arch[4],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[3], seed=3
            )
        ),
        layers.Dropout(actor_dropout),
        layers.Dense(
            actor_nn_arch[5],
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/actor_nn_arch[4], seed=4)
        )
    ])

    return actor_model


def create_critic_model():
    critic_model = keras.Sequential([
        layers.Input(shape=critic_nn_arch[0]),
        layers.Dense(
            critic_nn_arch[1],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[0], seed=11)
        ),
        layers.Dropout(critic_dropout),
        layers.Dense(
            critic_nn_arch[2],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[1], seed=12)
        ),
        layers.Dropout(critic_dropout),
        layers.Dense(
            critic_nn_arch[3],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[2], seed=13)
        ),
        layers.Dropout(critic_dropout),
        layers.Dense(
            critic_nn_arch[4],
            activation=leaky_relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[3], seed=13)
        ),
        layers.Dropout(critic_dropout),
        layers.Dense(
            critic_nn_arch[5],
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1/critic_nn_arch[4], seed=14)
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
        actions = np.argmax(model(df[[c for c in train.columns if "f_" in c] + ["feature_0", "weight"]].values).numpy(), axis=1)
        assert not np.isnan(np.sum(actions))

        sum_of_actions = np.sum(actions)
        print("np_sum(actions)", sum_of_actions)

    #     df["action"] = probs_df["action"]
        df["action"] = pd.Series(data=actions, index=df.index)
        df["trade_reward"] = df["action"]*df["weight"]*df["resp"]

        tmp = df.groupby(["date"])[["trade_reward"]].agg("sum")

        sum_of_pi = tmp["trade_reward"].sum()
        sum_of_pi_x_pi = (tmp["trade_reward"]*tmp["trade_reward"]).sum()

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
                                            shape=(1, self.observation_spec.shape[0]),
                                            dtype=self.observation_spec.dtype,

                                            ), name=self.observation_spec.name,
                                            trainable=False)
        self.prev_action = tf.Variable(
            0, dtype=np.int32, name="prev_action", trainable=False)

    def init(self, time_step):
        observation = time_step.observation

        probs = self.policy(observation)[0]

        action = tfp.distributions.Bernoulli(
            probs=probs[1], dtype=tf.int32).sample()

        self.prev_observation.assign(observation)
        self.prev_action.assign(action)

        return action

    def policy(self, observation):
        return tf.nn.softmax(self.actor_model(observation)/self.tau)

    def train(self, time_step):
        observation = time_step.observation

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
                 + self.critic_model(observation)[0][0]
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

        probs = self.policy(observation)[0]

        action = tfp.distributions.Bernoulli(
            probs=probs[1], dtype=tf.int32).sample()

        self.prev_action.assign(action)
        self.prev_observation.assign(observation)

        return action

# ### Agent Test


# +
# Test Cell

np.random.seed(42)


actor_model_test = keras.Sequential([
    layers.Input(shape=4),
    layers.Dense(
        4,
        activation="relu",
        kernel_initializer=tf.keras.initializers.Constant(value=1)
    ),
    layers.Dense(
        2,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.Constant(value=1)
    )
])


critic_model_test = keras.Sequential([
    layers.Input(shape=4),
    layers.Dense(
        4,
        activation="relu",
        kernel_initializer=tf.keras.initializers.Constant(value=1)
    ),
    layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.Constant(value=1)
    )
])

agent_test = ACAgent(
    actor_model=actor_model_test,
    critic_model=critic_model_test,
    avg_reward_step_size=0.1,
    actor_step_size=0.1,
    critic_step_size=0.1,
    observation_spec = tf.TensorSpec(shape = (4,), dtype=tf.float64, name="observation")
)
# agent_test.init = tf.function(agent_test.init, autograph = False, experimental_relax_shapes = True)

agent_test.train = tf.function(agent_test.train, autograph = False, experimental_relax_shapes = True)

# agent_test.update_td_error = tf.function(agent_test.update_td_error)


test_action = agent_test.init(TimeStep(
    step_type=tf.constant([0], dtype=np.int32, name="step_type"),
    reward=tf.constant([0.0], dtype=np.float32, name="reward"),
    discount=tf.constant([1], dtype=np.float32, name="discount"),
    observation=tf.constant(np.array([[1, 2, 1, 2]]), dtype=np.float64, name="observation")
    )
)
print("after init")
print(agent_test.actor_model.trainable_variables)

# print("test_action: ________" , test_action)
# assert test_action.numpy() == 0

# agent_test.update_avg_reward(1)
# assert agent_test.reward == 0.1
# agent_test.update_avg_reward(1)
# assert agent_test.reward == 0.19
# agent_test.update_avg_reward(1)
# assert agent_test.reward == 0.271

# agent_test.update_td_error(1, tf.constant(
#     np.array([[1, 1, 0, 0]]), dtype=np.float64))


# print("delta: _______", agent_test.delta.numpy())
# assert agent_test.delta.numpy() == -7.271000000000001

test_action = agent_test.train(TimeStep(
    step_type=tf.constant([0], dtype=np.int32, name="step_type"),
    reward=tf.constant([5.0], dtype=np.float32, name="reward"),
    discount=tf.constant([1], dtype=np.float32, name="discount"),
    observation=tf.constant([[1, 2, 1, 2]], dtype=np.float64, name="observation"))
)

print("after Train")

print(agent_test.actor_model.trainable_variables)

test_action = agent_test.train(TimeStep(
    step_type=tf.constant([0], dtype=np.int32, name="step_type"),
    reward=tf.constant([3.0], dtype=np.float32, name="reward"),
    discount=tf.constant([1], dtype=np.float32, name="discount"),
    observation=tf.constant([[1, 2, 1, 2]], dtype=np.float64, name="observation"))
)

print("after Train")

print(agent_test.actor_model.trainable_variables)


# -


# ## Running of the experiment

# ### Running Full Experiment

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
    
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        best_score = 0
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
                    
                    if u_eval > best_score:
                        best_score = u_eval
                        agent.actor_model.save("./actor_model") 
                        
                    mlflow.log_metrics({
                        "t_eval": t_eval,
                        "u_eval": u_eval,
                        "t_train": t_train,
                        "u_train": u_train,
                        "ratio_of_ones_eval": ratio_of_ones_eval,
                        "ratio_of_ones_train": ratio_of_ones_train
                    })
                
        subprocess.run(["zip", "-r", "model.zip", "actor_model"])
        mlflow.log_artifact("model.zip")


run_experiment()
# -
# ##### Debugging

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


