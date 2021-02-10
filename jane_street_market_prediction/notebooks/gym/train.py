import pandas as pd
from optuna.pruners import SuccessiveHalvingPruner
import optuna
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf

from optuna.samplers import RandomSampler, TPESampler
from optuna import Trial

import logging
import sys

from callbacks.TrialCallback import TrialEvalCallback
from envs.gym_market_env import MarketEnvDaily

train = pd.read_csv("../etl/train_dataset_after_pca.csv")
eval_df = pd.read_csv("../etl/val_dataset_after_pca.csv")

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "ppo2-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

def run(n_timesteps = train.shape[0], seed=42, n_trials = 100):

    n_startup_trials = 10
    # evaluate every 20th of the maximum budget per iteration
    n_evaluations = 20
    eval_freq = int(n_timesteps / n_evaluations)

    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )

    def param_sampler(trial:Trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical('gamma', [0.001, 0.01, 0.1, 0.2, 0.3, 0.5])
        learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
        ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
        cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
        noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
        lam = trial.suggest_categorical('lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

        if n_steps < batch_size:
            nminibatches = 1
        else:
            nminibatches = int(n_steps / batch_size)

        multiplicator_coef = trial.suggest_uniform("multiplicator_coef", 0.8, 1.2)
        reward_multiplicator = 100
        negative_reward_multiplicator = reward_multiplicator * multiplicator_coef

        features = [c for c in train.columns.values if "f_" in c] + ["feature_0", "weight"]

        train_py_env = MarketEnvDaily(
            trades=train,
            features=features,
            reward_column="resp",
            weight_column="weight",
            include_weight=True,
            reward_multiplicator=reward_multiplicator,
            negative_reward_multiplicator=negative_reward_multiplicator
        )

        train_env = DummyVecEnv([lambda: train_py_env])

        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        net_arch = []
        for i in range(num_layers):
            l = trial.suggest_categorical("layer_{i}".format(i=i), [1, 2, 3])
            net_arch.append(l*len(features))


        policy_kwargs = dict(act_fun=tf.nn.swish, net_arch=net_arch)

        return {
            'policy': MlpPolicy,
            'env': train_env,
            'n_steps': n_steps,
            'nminibatches': nminibatches,
            'gamma': gamma,
            'learning_rate': learning_rate,
            'ent_coef': ent_coef,
            'cliprange': cliprange,
            'noptepochs': noptepochs,
            'policy_kwargs': policy_kwargs,
            'lam': lam
        }
    def objective(trial):
        kwargs = param_sampler(trial)
        model = PPO2(**kwargs)
        eval_callback = TrialEvalCallback(train, eval_df, trial, eval_freq=eval_freq)

        try:
            model.learn(n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            raise optuna.exceptions.TrialPruned()

        is_pruned = eval_callback.is_pruned
        sum_of_t_coef = -1 * eval_callback.sum_of_t_coef

        del model.env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return sum_of_t_coef

    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == "__main__":
    run()