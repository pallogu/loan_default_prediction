import gym
from gym import spaces

import pandas as pd
import numpy as np

class MarketEnvDaily(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super(MarketEnvDaily, self).__init__()
        
        self.trades = kwargs.get("trades")
        self.features = kwargs.get("features")
        self.reward_column = kwargs.get("reward_column")
        self.weight_column = kwargs.get("weight_column")
        self.reward_multiplicator = kwargs.get("reward_multiplicator", 1)
        self.negative_reward_multiplicator = kwargs.get("negative_reward_multiplicator", 1)
        self.include_weight = kwargs.get("include_weight", True)
        self.unique_dates = self.trades["date"].unique()

        self.selected_date = np.random.choice(self.unique_dates)
        self.active_day_trades = self.trades[self.trades["date"] == self.selected_date]

        self.intraday_counter = 0

        self._episode_ended = False
        
        self.action_space = spaces.Discrete(2)
        # Example for using image as input (can be channel-first or channel-last):
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.features),), dtype=np.float32)

    def step(self, action):
        if self.intraday_counter == (len(self.active_day_trades) - 2):
            self._episode_ended = True

        observation = self.active_day_trades.iloc[self.intraday_counter + 1][self.features].values


        reward = 0 if action == 0 else self.active_day_trades.iloc[self.intraday_counter][self.reward_column]*self.active_day_trades.iloc[self.intraday_counter][self.weight_column]
        
        if reward > 0:
            reward = reward * self.reward_multiplicator

        if reward < 0:
            reward = reward * self.negative_reward_multiplicator

        if self._episode_ended:
            done = True
        else:
            done = False
            
        info = {
            "day": self.selected_date,
            "reward": reward
        }

        self.intraday_counter += 1
        return observation, reward, done, info
    
    def reset(self):
        self.intraday_counter = 0
        self.selected_date = np.random.choice(self.unique_dates)
        self.active_day_trades = self.trades[self.trades["date"] == self.selected_date]

        observation = np.array(self.active_day_trades.iloc[self.intraday_counter][self.features].values, dtype=np.float32)
        self._episode_ended = False
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass
        
    def close (self):
        pass


class MarketEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super(MarketEnv, self).__init__()
        
        self.trades = kwargs.get("trades")
        self.features = kwargs.get("features")
        self.reward_column = kwargs.get("reward_column")
        self.weight_column = kwargs.get("weight_column")
        self.reward_multiplicator = kwargs.get("reward_multiplicator", 1)
        self.negative_reward_multiplicator = kwargs.get("negative_reward_multiplicator", 1)
        self.include_weight = kwargs.get("include_weight", True)

        self.counter = 0

        self._episode_ended = False
        
        self.action_space = spaces.Discrete(2)
        # Example for using image as input (can be channel-first or channel-last):
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(self.features),), dtype=np.float32)

    def step(self, action):
        if self.counter == (len(self.trades) - 2):
            self._episode_ended = True

        observation = self.trades.iloc[self.counter + 1][self.features].values

        if self.include_weight:
            reward = 0 if action == 0 else self.trades.iloc[self.counter][self.reward_column]*self.trades.iloc[self.counter][self.weight_column]
        else:
            reward = 0 if action == 0 else self.trades.iloc[self.counter][self.reward_column]

        if reward > 0:
            reward = reward * self.reward_multiplicator

        if reward < 0:
            reward = reward * self.negative_reward_multiplicator

        if self._episode_ended:
            done = True
        else:
            done = False
            
        info = {
            "reward": reward
        }

        self.counter += 1
        return observation, reward, done, info
    
    def reset(self):
        self.counter = 0

        observation = np.array(self.trades.iloc[self.counter][self.features].values, dtype=np.float32)
        self._episode_ended = False
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass
        
    def close (self):
        pass
