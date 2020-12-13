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

# -



def evaluate_policy(agent, valuation_data, feats):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    print("evaluating policy")
    valuation = {}
    
    
    def valuation_data_gen(valuation_data):
        counter = 0
        while counter < valuation_data.shape[0]:
            yield valuation_data.iloc[counter]
            counter += 1

    gen = valuation_data_gen(valuation_data)
    
    for observation  in gen:
        obser_feat = observation[feats]
        reward = observation["resp"]
        weight = observation["weight"]
        date = observation["date"]
        action = agent.policy(feats)
        
        try:
            valuation[date]
        except KeyError:
            valuation[date] = 0

        valuation[date] += reward*action*weight
        
    sum_of_pi = 0
    sum_of_pi_x_pi = 0
    
    for value in valuation.values():
        sum_of_pi += value
        sum_of_pi_x_pi += value*value
        
    t = sum_of_pi/np.sqrt(sum_of_pi_x_pi) * np.sqrt(250/len(valuation.values()))
    
    u  = np.max([np.min([t, 0]), 6]) * sum_of_pi
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    print("evaluating policy")
        
    return u

evaluate_policy(random_policy_agent, valuation_data, feats)


# +
def run_experiment(number_of_iterations, agent):
    
#     pool = mp.Pool(mp.cpu_count() -1 )
    
#     valuations = [pool.apply(evaluate_policy, args=(agent, valuation_data, feats)) for i in range(number_of_iterations)]

    valuations = [evaluate_policy(agent, valuation_data, feats) for i in range(number_of_iterations)]

#     pool.close()
    
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

run_experiment(100, random_policy_agent)
# -


