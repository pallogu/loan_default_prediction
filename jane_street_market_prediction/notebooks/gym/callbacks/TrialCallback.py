from stable_baselines.common.callbacks import BaseCallback, EvalCallback

from callbacks.common import calculate_u_metric


class TrialEvalCallback(BaseCallback):
    """
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, train_df, eval_df, trial, eval_freq=10000,verbose=0):

        super(TrialEvalCallback, self).__init__(verbose=verbose)
        self.trial = trial
        self.eval_df = eval_df
        self.train_df = train_df
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.sum_of_t_coef = -100
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            t_eval, u_eval, ratio_of_ones_eval = calculate_u_metric(self.eval_df, self.model, verbose=1)
            t_train, u_train, ratio_of_ones_train = calculate_u_metric(self.train_df, self.model, verbose=1)

            self.sum_of_t_coef = t_eval + t_train
            self.trial.report(-1 * (self.sum_of_t_coef), self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True