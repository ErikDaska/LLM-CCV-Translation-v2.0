"""Bridge epoch-based Hugging Face evaluations to Optuna pruning."""

import math

import optuna
from transformers import TrainerCallback


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.trial.Trial, metric_name: str = "eval_chrf"):
        self.trial = trial
        self.metric_name = metric_name
        self.reported_epochs = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        if self.metric_name not in metrics or state.epoch is None:
            return

        # Designed for eval_strategy='epoch'. Epoch count is comparable between trials.
        epoch = int(round(float(state.epoch)))
        if epoch < 1 or epoch in self.reported_epochs:
            return

        value = float(metrics[self.metric_name])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite pruning metric {self.metric_name}: {value}")

        self.reported_epochs.add(epoch)
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned trial {self.trial.number} at epoch {epoch}: {self.metric_name}={value:.4f}")