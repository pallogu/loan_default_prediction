# +
import tensorflow as tf
import numpy as np

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
        reward = 0 if action == 0 else self.trades.iloc[self.counter][self.reward_column]*self.trades.iloc[self.counter][self.weight_column]
        
        
        if self._episode_ended:
            time_step = ts.termination(np.array(self._state, dtype = np.float64), reward)
        else:
            time_step = ts.transition(np.array(self._state, dtype = np.float64), reward, discount = self.discount)
        

            
        self.counter += 1
        
        return time_step

env = MarketEnv(
    trades=trades,
    features=features,
    reward_column = reward_column,
    weight_column = "weight"
)

utils.validate_py_environment(env, episodes=5)

env.reset()

env.step(1)


