import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy.random import seed
from tensorflow.random import set_seed

seed(42)
set_seed(42)

# ### Environments

from environment import MarketEnv

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

train.head()

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

time_step = tf_env.reset()

time_step.observation

tf_env.time_step_spec().reward

tf_env.action_spec()

actor_nn_arch = (
    tf_env.time_step_spec().observation.shape[0],
    32,
    32,
    2
)
actor_model = keras.Sequential([
    layers.Input(shape = actor_nn_arch[0]),
    layers.Dense(
        actor_nn_arch[1],
        activation = "relu",
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/actor_nn_arch[0], seed=1)
    ),
    layers.Dense(
        actor_nn_arch[2],
        activation = "relu",
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/actor_nn_arch[1], seed=2)
    ),
    layers.Dense(
        actor_nn_arch[3],
        activation = "softmax",
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/actor_nn_arch[2], seed=3)
    )
])

critic_nn_arch = (
    tf_env.time_step_spec().observation.shape[0],
    32,
    32,
    1
)

critic_model = keras.Sequential([
    layers.Input(shape = critic_nn_arch[0]),
    layers.Dense(
        critic_nn_arch[1],
        activation = "relu",
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/critic_nn_arch[0], seed=11)
    ),
    layers.Dense(
        critic_nn_arch[2],
        activation = "relu",
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/critic_nn_arch[1], seed=12)
    ),
    layers.Dense(
        critic_nn_arch[3],
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/critic_nn_arch[2], seed=13)
    )
])

np.argmax(actor_model(eval_df[["f_{i}".format(i=i) for i in range(40)] + ["weight"]].values). numpy(), axis=1)





# ## Hyperparameters

avg_reward_step_size = 0.2
actor_step_size = 0.5
critic_step_size = 0.5
number_of_episodes = 10


# ## AC Agent

class ACAgent():
    def __init__(self, **kwargs):
        self.actor_model = kwargs.get("actor_model")
        self.critic_model = kwargs.get("critic_model")
        self.avg_reward_step_size = kwargs.get("avg_reward_step_size")
        
        actor_step_size = kwargs.get("actor_step_size")
        critic_step_size = kwargs.get("critic_step_size")
        
        self.actor_optimizers = [
            keras.optimizers.Adam(learning_rate=actor_step_size),
            keras.optimizers.Adam(learning_rate=actor_step_size)
        ]
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_step_size)
        
        self.reward = None
        self.delta = None
        
        
    def train(self, time_step):
        observation = time_step.observation
        reward = time_step.reward
        
        self.update_avg_reward(reward)
        self.update_td_error(reward, observation)
        self.update_critic_model(observation)
        self.update_actor_model(observation)
        
        action = np.random.choice(
            [0, 1],
            p=self.actor_model(observation).numpy()[0]
        )
        
        self.prev_action = action
        self.prev_observation = observation
        
        return action
        
    def update_avg_reward(self, reward):
        self.reward += self.avg_reward_step_size * reward
        
    def update_td_error(self, reward, observation):
        self.delta = (
            reward 
            - self.reward
            + self.critic_model(observation).numpy()[0][0]
            - self.critic_model(self.prev_observation).numpy[0][0]
        )
        
    def update_critic_model(self, observation):
        with tf.GradientTape() as tape:
            grad = [-1 * self.delta * g for g in  tape.gradient(
                self.critic_model(observation),
                self.critic_model.trainable_variables
            )]
            
            self.critic_optimizer.apply_gradients(
                zip(grad, self.critic_model.trainable_variables)
            )
            
    def update_actor_model(self, observation):
        prev_action = self.prev_action
        with tf.GradientTape() as tape:
            grad = [-1 * self.delta * g for g in tape.gradient(
                tf.math.log(self.actor_model(observation)),
                self.actor_model.trainable_variables
            )]
            
            last_layer_w = grad[-2].numpy()
            last_layer_w[:, prev_action] = 0
            grad[-2] =  tf.constant(last_layer_w, dtype=np.float32)
            
            last_layer_b = grad[-1].numpy()
            last_layer_b[prev_action] = 0
            grad[-1] =  tf.constant(last_layer_b, dtype=np.float32)
            
            
            self.actor_optimizers[prev_action].apply_gradients(
                zip(grad, self.actor_model.trainable_variables)
            )        

while not time_step.is_last():


# +
def run_experiment():
    
        
# -

tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)




# +
def train_eval(

    # Evaluation params

    
):
    
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes)
        tf_metrics.AverageEpisodeLengthMetric=num_eval_episodes
    ]
    
    
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]
    
    tf_agent.train = common.function(tf_agent.train)
    
    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    if use_tf_functions:
        train_step = common.function(train_step)
        
    for _ in range(num_iterations):
        for _ in range(episode_steps):
            for _ in range(train_steps_per_iteration):
                train_step()
