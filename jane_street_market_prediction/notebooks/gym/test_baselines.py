# %load_ext autoreload
# %autoreload 2

# +
import pandas as pd
import numpy as np
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.callbacks import BaseCallback

from envs.gym_market_env import MarketEnv
import mlflow

import tensorflow as tf
# -

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

# +
reward_multiplicator = 100
negative_reward_multiplicator = 100

train_py_env = MarketEnv(
    trades = train,
    features = [c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"],
    reward_column = "resp",
    weight_column = "weight",
    include_weight=True,
    reward_multiplicator = reward_multiplicator,
    negative_reward_multiplicator = negative_reward_multiplicator
)

eval_py_env = MarketEnv(
    trades = train,
    features = [c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"],
    reward_column = "resp",
    weight_column = "weight",
    include_weight=True,
    reward_multiplicator = 1,
    negative_reward_multiplicator = 1
)


# +
class EvaluationCallback(BaseCallback):

    
    def __init__(self, verbose=0, eval_df=None, train_df=None, log_interval=10000):
        super(EvaluationCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        
        self.verbose = verbose
        self.eval_df = eval_df
        self.train_df = train_df
        self.log_interval = log_interval
        self.step = 0
        

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.step += 1
        if self.step % self.log_interval == 0:
            t_eval, u_eval, ratio_of_ones_eval = self.calculate_u_metric(self.eval_df)
            t_train, u_train, ratio_of_ones_train = self.calculate_u_metric(self.train_df)
            
            mlflow.log_metrics({
                    "t_eval": t_eval,
                    "u_eval": u_eval,
                    "t_train": t_train,
                    "u_train": u_train,
                    "ratio_of_ones_eval": ratio_of_ones_eval,
                    "ratio_of_ones_train": ratio_of_ones_train
                }, step=self.step)
            
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    
    def calculate_u_metric(self, df):
        actions = self.model.predict(df[[c for c in df.columns if "f_" in c] + ["feature_0","weight"]].values, deterministic=True)[0]
        assert not np.isnan(np.sum(actions))

        sum_of_actions = np.sum(actions)


        df["action"] = pd.Series(data=actions, index=df.index)

        df["trade_reward"] = df["action"]*df["weight"]*df["resp"]

        tmp = df.groupby(["date"])[["trade_reward"]].agg("sum")

        sum_of_pi = tmp["trade_reward"].sum()
        sum_of_pi_x_pi = (tmp["trade_reward"]*tmp["trade_reward"]).sum()

        t = sum_of_pi/np.sqrt(sum_of_pi_x_pi) * np.sqrt(250/tmp.shape[0])
        u  = np.min([np.max([t, 0]), 6]) * sum_of_pi
        ratio_of_ones = sum_of_actions/len(actions)
        
        if self.verbose == 1:
            print("sum of pi: {sum_of_pi}".format(sum_of_pi = sum_of_pi) )
            print("t: {t}".format(t = t) )
            print("u: {u}".format(u = u) )
            print("np_sum(actions)", sum_of_actions)
            print("ration of ones", ratio_of_ones)
            print("length of df", len(actions))

        return t, u, ratio_of_ones
        
    
# -

train_env = DummyVecEnv([lambda: train_py_env])
evaluation_callback = EvaluationCallback(verbose=0, eval_df=eval_df, train_df=train, log_interval=10000)

# %%time
with mlflow.start_run():
    policy_kwargs = dict(act_fun=tf.nn.swish, net_arch=[128, 128, 64, 32])
    gamma=0.01
    learning_rate=1e-6
    
    mlflow.set_tag("agent_type", "PPO")
    mlflow.log_param("policy", "mlp")
    mlflow.log_param("policy_kwargs", policy_kwargs)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("learning_rate", learning_rate)
#     model = PPO2(MlpPolicy, train_env, verbose=0, gamma=gamma, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10000000, callback=evaluation_callback)

model.predict(eval_df[[c for c in eval_df.columns if "f_" in c] + ["feature_0","weight"]].values, deterministic=True)[0]


