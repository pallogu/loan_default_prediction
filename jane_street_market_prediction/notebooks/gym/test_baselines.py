# %load_ext autoreload
# %autoreload 2

import mlflow
# +
import pandas as pd
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from callbacks.EvaluationCallback import EvaluationCallback
from envs.gym_market_env import MarketEnv

# -

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

# +
reward_multiplicator = 100
negative_reward_multiplicator = 100*1.0582972011771625

features = [c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"]

train_py_env = MarketEnv(
    trades=train,
    features=features,
    reward_column="resp",
    weight_column="weight",
    include_weight=True,
    reward_multiplicator=reward_multiplicator,
    negative_reward_multiplicator=negative_reward_multiplicator
)
# -

train_env = DummyVecEnv([lambda: train_py_env])
evaluation_callback = EvaluationCallback(verbose=0, eval_df=eval_df, train_df=train, log_interval=10000, save_best=True)

# %%time
with mlflow.start_run():
    num_features =len(features)
    policy_kwargs = dict(act_fun=tf.nn.swish, net_arch=[3*num_features, 2*num_features, num_features])

    
    kwargs = {
        "policy": MlpPolicy,
        "env": train_env,
        "learning_rate": 0.0001947227876477586,
        "gamma": 0.2,
        "nminibatches": int(2048/128),
        "noptepochs": 5,
        "ent_coef": 0.0005567548347339001,
        "n_steps": 2048,
        "lam": 0.9,
        "cliprange":0.1,
        "policy_kwargs": policy_kwargs
    }

    mlflow.set_tag("agent_type", "PPO")
    mlflow.log_params(kwargs)

    model = PPO2(**kwargs)
    model.learn(total_timesteps=train.shape[0] * 10, callback=evaluation_callback)


