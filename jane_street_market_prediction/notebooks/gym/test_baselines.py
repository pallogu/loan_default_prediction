# %load_ext autoreload
# %autoreload 2

import mlflow
# +
import pandas as pd
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from notebooks.gym.callbacks.EvaluationCallback import EvaluationCallback
from .envs.gym_market_env import MarketEnv

# -

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

# +
reward_multiplicator = 100
negative_reward_multiplicator = 100

train_py_env = MarketEnv(
    trades=train,
    features=[c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"],
    reward_column="resp",
    weight_column="weight",
    include_weight=True,
    reward_multiplicator=reward_multiplicator,
    negative_reward_multiplicator=negative_reward_multiplicator
)

eval_py_env = MarketEnv(
    trades=train,
    features=[c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"],
    reward_column="resp",
    weight_column="weight",
    include_weight=True,
    reward_multiplicator=1,
    negative_reward_multiplicator=1
)

# +


# -

train_env = DummyVecEnv([lambda: train_py_env])
evaluation_callback = EvaluationCallback(verbose=0, eval_df=eval_df, train_df=train, log_interval=10000, save_best=True)

# %%time
with mlflow.start_run():
    policy_kwargs = dict(act_fun=tf.nn.swish, net_arch=[128, 128, 64, 32])
    gamma = 0.01
    learning_rate = 1e-6

    mlflow.set_tag("agent_type", "PPO")
    mlflow.log_param("policy", "mlp")
    mlflow.log_param("policy_kwargs", policy_kwargs)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("learning_rate", learning_rate)
    model = PPO2(MlpPolicy, train_env, verbose=0, gamma=gamma, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=train.shape[0] * 10, callback=evaluation_callback)
