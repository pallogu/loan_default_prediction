# %load_ext autoreload
# %autoreload 2

# +
from numpy.core.numeric import outer
import tensorflow as tf
import numpy as np
from tensorflow.python.util.tf_export import kwarg_only

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec

from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

import pandas as pd

# +
data = [[ 1.00000000e+00, -1.87274634e+00, -2.19124240e+00,
         6.27036224e-03,  0.1, 0],
       [-1.00000000e+00, -1.34953705e+00, -1.70470899e+00,
          -9.79168235e-03,  0, 1],
       [-1.00000000e+00,  8.12780428e-01, -2.56155843e-01,
         2.39701263e-02,  0.1, 2],
       [-1.00000000e+00,  1.17437847e+00,  3.44640088e-01,
         -3.20009790e-03,  0.1, 3],
       [ 1.00000000e+00, -3.17202633e+00, -3.09318225e+00,
         -2.60357587e-03,  0.2, 4],
       [-1.00000000e+00, -1.49968085e+00, -1.92802273e+00,
         1.28169863e-03,  0.2, 5],
       [-1.00000000e+00, -3.17202633e+00, -3.09318225e+00,
         7.09105293e-04,  0, 6],
       [-1.00000000e+00,  4.46049929e-01, -4.66209757e-01,
         3.34726094e-02,  0.1, 7],
       [ 1.00000000e+00, -3.17202633e+00, -3.09318225e+00,
         -1.67740650e-03,  0.1, 8],
       [ 1.00000000e+00,  2.74440750e+00,  1.41212671e+00,
         2.03169999e-02,  0.1, 9]]

features = ['feature_0', 'feature_1', 'feature_2']
reward_column = "resp"

trades = pd.DataFrame(data = data, columns=['feature_0', 'feature_1', 'feature_2', 'resp', 'weight', 'date'])
# -

trades


class MarketEnv(py_environment.PyEnvironment):
    def __init__(self, **kwargs):
        
        self.trades = kwargs.get("trades")
        self.features = kwargs.get("features")
        self.reward_column = kwargs.get("reward_column")
        self.weight_column = kwargs.get("weight_column")
        self.discount = kwargs.get("discount", 1)
        self.reward_multiplicator = kwargs.get("reward_multiplicator", 1)
        self.negative_reward_multiplicator = kwargs.get("negative_reward_multiplicator", 1)
        self.include_weight = kwargs.get("include_weight", True)
        
        self.counter = 0
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=1,
            name="action"
        )
        
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self.features), ),
            dtype=np.float64,
            name='observation'
        )
        
        self._state = self.trades.iloc[self.counter][self.features].values

        
        self._episode_ended = False
        
        
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self.counter = 0
        self._state = self.trades.iloc[self.counter][self.features].values
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _step(self, action):
        
        
        if self._episode_ended:
            return self.reset()
        
        if self.counter == (len(self.trades) - 2):
            self._episode_ended = True
            
        self._state = self.trades.iloc[self.counter + 1][self.features].values
        
        if self.include_weight:
            reward = 0 if action == 0 else self.trades.iloc[self.counter][self.reward_column]*self.trades.iloc[self.counter][self.weight_column]
        else:
            reward = 0 if action == 0 else self.trades.iloc[self.counter][self.reward_column]
        
        if reward > 0:
            reward = reward*self.reward_multiplicator
            
        if reward < 0:
            reward = reward*self.negative_reward_multiplicator
        
        if self._episode_ended:
            time_step = ts.termination(np.array(self._state, dtype = np.float64), np.array(reward, dtype=np.float64))
        else:
            time_step = ts.transition(np.array(self._state, dtype = np.float64), np.array(reward, dtype=np.float64), discount=self.discount)
        

            
        self.counter += 1
        
        return time_step


class MarketEnvContinuous(py_environment.PyEnvironment):
    def __init__(self, **kwargs):
        
        self.trades = kwargs.get("trades")
        self.features = kwargs.get("features")
        self.reward_column = kwargs.get("reward_column")
        self.weight_column = kwargs.get("weight_column")
        self.discount = kwargs.get("discount", 1)
        self.reward_multiplicator = kwargs.get("reward_multiplicator", 1)
        self.negative_reward_multiplicator = kwargs.get("negative_reward_multiplicator", 1)
        self.include_weight = kwargs.get("include_weight", True)
        self.unique_dates = self.trades["date"].unique()
        
        self.selected_date = np.random.choice(self.unique_dates)
        self.active_day_trades = self.trades[self.trades["date"] == self.selected_date]
        
        self.counter = 0
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.float32,
            minimum=0,
            maximum=1,
            name="action"
        )
        
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self.features), ),
            dtype=np.float32,
            name='observation'
        )
        
        self._state = np.array(self.active_day_trades.iloc[self.counter][self.features].values, dtype = np.float32)

        
        self._episode_ended = False
        
        
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self.counter = 0
        self.selected_date = np.random.choice(self.unique_dates)
        self.active_day_trades = self.trades[self.trades["date"] == self.selected_date]
        
        self._state = np.array(self.active_day_trades.iloc[self.counter][self.features].values, dtype = np.float32)
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _step(self, action):
        
        
        if self._episode_ended:
            return self.reset()
        
        if self.counter == (len(self.active_day_trades) - 2):
            self._episode_ended = True
            
        self._state = self.active_day_trades.iloc[self.counter + 1][self.features].values
        
        if self.include_weight:
            reward = 0 if action < 0.5 else self.active_day_trades.iloc[self.counter][self.reward_column]*self.active_day_trades.iloc[self.counter][self.weight_column]
        else:
            reward = 0 if action < 0.5 else self.active_day_trades.iloc[self.counter][self.reward_column]
        
        if reward > 0:
            reward = reward*self.reward_multiplicator
            
        if reward < 0:
            reward = reward*self.negative_reward_multiplicator
        
        if self._episode_ended:
            time_step = ts.termination(np.array(self._state, dtype = np.float32), np.array(reward, dtype=np.float32))
        else:
            time_step = ts.transition(np.array(self._state, dtype = np.float32), np.array(reward, dtype=np.float32), discount=float(self.discount))
        

            
        self.counter += 1
        
        return time_step


class MarketEnvSimplified(py_environment.PyEnvironment):
    def __init__(self, **kwargs):
        
        self.trades = kwargs.get("trades")
        self.features = kwargs.get("features")
        self.reward_column = kwargs.get("reward_column")
        self.discount = kwargs.get("discount", 1)
        
        self.counter = 0
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=1,
            name="action"
        )
        
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self.features), ),
            dtype=np.float64,
            name='observation'
        )
                
        self._state = self.trades.iloc[self.counter][self.features].values

        
        self._episode_ended = False
        
        
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self.counter = 0

        self._state = self.trades.iloc[self.counter][self.features].values
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _step(self, action):
        
        
        if self._episode_ended:
            return self.reset()
        
        if self.counter == (len(self.trades) - 2):
            self._episode_ended = True
            
        self._state = self.trades.iloc[self.counter + 1][self.features].values
        
        if action == 0:
            reward = 0
        else:
            if self.trades.iloc[self.counter][self.reward_column] > 0:
                reward = 1
            elif self.trades.iloc[self.counter][self.reward_column] < 0:
                reward = -1
            else:
                reward = 0
        
        if self._episode_ended:
            time_step = ts.termination(np.array(self._state, dtype = np.float64), np.array(reward, dtype=np.float64))
        else:
            time_step = ts.transition(np.array(self._state, dtype = np.float64), np.array(reward, dtype=np.float64), discount=self.discount)
        

            
        self.counter += 1
        
        return time_step

env = MarketEnv(
    trades=trades,
    features=features,
    reward_column = reward_column,
    weight_column = "weight",
    reward_multiplicator = 100, 
    negative_reward_multiplicator = 10000
)

utils.validate_py_environment(env, episodes=5)

env.reset()

env.step(1)


class MarketEnvWithRiskAppetite(py_environment.PyEnvironment):
    def __init__(self, **kwargs):

        self.trades = kwargs.get("trades")
        self.features = kwargs.get("features")
        self.reward_column = kwargs.get("reward_column")
        self.weight_column = kwargs.get("weight_column")
        self.discount = kwargs.get("discount", 1)
        self.risk_apetite = kwargs.get("risk_apetite", -100)
        self.out_of_funds_penalty = kwargs.get("out_of_funds_penalty", 500)

        self.counter = 0
        self.pnl = 0

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=1,
            name="action"
        )

        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self.features), ),
            dtype=np.float64,
            name='observation'
        )

        self._state = self.trades.iloc[self.counter][self.features].values

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.counter = 0
        self.pnl = 0
        self.trades = self.trades.sample(frac=1.0)
        self._state = self.trades.iloc[self.counter][self.features].values
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        out_of_funds = self.pnl < self.risk_apetite

        if self._episode_ended:
            return self.reset()

        if self.counter == (len(self.trades) - 2):
            self._episode_ended = True

        if out_of_funds:
            self._episode_ended = True

        self._state = self.trades.iloc[self.counter + 1][self.features].values
        reward = 0 if action == 0 else self.trades.iloc[self.counter][self.reward_column] * \
            self.trades.iloc[self.counter][self.weight_column]
        self.pnl += reward

        if out_of_funds:
            reward -= self.out_of_funds_penalty

        if self._episode_ended:
            time_step = ts.termination(
                np.array(self._state, dtype=np.float64), reward)
        else:
            time_step = ts.transition(
                np.array(self._state, dtype=np.float64), reward, discount=self.discount)

        self.counter += 1

        return time_step


env_with_risk_appetite = MarketEnvWithRiskAppetite(
    trades=trades,
    features=features,
    reward_column = reward_column,
    weight_column = "weight",
    risk_apetite = 0,
    out_of_funds_penalty = 100
)

utils.validate_py_environment(env_with_risk_appetite, episodes=5)

env_with_risk_appetite.reset()

time_step = env_with_risk_appetite.step(1)
print(time_step.is_last())
print(time_step)




