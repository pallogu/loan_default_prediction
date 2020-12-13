# ## Baseline model evaluation

from environment import MarketEnv
import pandas as pd
import numpy as np
import mlflow
import multiprocessing as mp
import time

train = pd.read_csv("../../input/train.csv")

train_data = train[train["date"] < 400]
valuation_data = train[train["date"] >=400]

feats = ["feature_{count}".format(count = count) for count in range(0, 130)]


# +
class RandomPolicyAgent():
    def policy(self, features):
        return np.random.choice([0,1])

random_policy_agent = RandomPolicyAgent()


# +
class AlwaysBuyAgent():
    def policy(self, features):
        return 1

always_buy_agent = AlwaysBuyAgent()

# -

temp = pd.DataFrame(data=[
    {
        "date": 0,
        "weight": 1,
        "resp": 1
    },
    {
        "date": 0,
        "weight": 1,
        "resp": 2
    },
    {
        "date": 1,
        "weight": 1,
        "resp": 1
    }
], columns=["date", "trade_reward"])


def evaluate_policy(agent, valuation_data, feats):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("start_time", current_time)
    val_data = valuation_data.copy()
    
    val_data["action"] = val_data[feats].apply(lambda row: agent.policy(row), axis=1)
    val_data["trade_reward"] = val_data["action"]*val_data["weight"]*val_data["resp"]
    val_data["trade_reward_squared"] = val_data["trade_reward"]*val_data["trade_reward"]

    tmp = val_data.groupby(["date"])[["trade_reward", "trade_reward_squared"]].agg("sum")
        
    sum_of_pi = tmp["trade_reward"].sum()
    sum_of_pi_x_pi = tmp["trade_reward_squared"].sum()
        
    t = sum_of_pi/np.sqrt(sum_of_pi_x_pi) * np.sqrt(250/tmp.shape[0])
    
    u  = np.max([np.min([t, 0]), 6]) * sum_of_pi
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("end_time", current_time)
        
    return u


# +
def run_experiment(number_of_iterations, agent):
    
    valuations = [evaluate_policy(agent, valuation_data, feats) for i in range(number_of_iterations)]
    
    with mlflow.start_run():
        
        mlflow.set_tag("agent_type", "random")
        mlflow.log_param("p", 0.5)
        mlflow.log_param("number of iterations", number_of_iterations)
        
        val_np = np.array(valuations)

        mean = val_np.mean()
        std_dev = val_np.std()
        
        mlflow.log_metric("mean", mean)
        mlflow.log_metric("std dev", std_dev)

    return mean, std_dev

run_experiment(1000, random_policy_agent)
# -


