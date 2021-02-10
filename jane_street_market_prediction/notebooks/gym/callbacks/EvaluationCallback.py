from stable_baselines.common.callbacks import BaseCallback
import mlflow
import subprocess

from common import calculate_u_metric

class EvaluationCallback(BaseCallback):
    def __init__(self, verbose=0, eval_df=None, train_df=None, log_interval=10000, save_best=True):
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
        self.best_u_eval = 0

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
            t_eval, u_eval, ratio_of_ones_eval = calculate_u_metric(self.eval_df, self.model, verbose=1)
            t_train, u_train, ratio_of_ones_train = calculate_u_metric(self.train_df, self.model, verbose=1)

            if u_eval > self.best_u_eval:
                self.best_u_eval = u_eval
                self.model.save("./model")

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

        # subprocess.run(["zip", "-r", "model.zip", "magic"])
        mlflow.log_artifact("model.zip")
        pass
