"""Run sequential Optuna trials using the shared training implementation."""

import argparse
from copy import deepcopy
import gc
import logging
import math

# Importa primeiro o módulo que configura o ambiente do treino.
from train import TrainingTranslationScript
from utils.configuration import load_training_configuration

import optuna
import torch

logger = logging.getLogger(__name__)


class HPOTrainingScript:
    """Manage an Optuna study and execute an independent training per trial."""

    def __init__(self, config_path):
        self.config = load_training_configuration(config_path)
        self.hpo = self.config["hpo"]

        if "training_arguments" not in self.config:
            raise ValueError("O YAML deve usar 'training_arguments', não 'training'.")

        if not self.hpo.get("search_space"):
            raise ValueError("O espaço de pesquisa está vazio.")

        for field in ("study_name", "storage", "objective", "direction", "sampler_seed", "trials"):
            if field not in self.hpo:
                raise ValueError(f"Falta hpo.{field} na configuração.")

    def _sample_parameters(self, trial):
        """Sample the distributions declared in YAML."""
        sampled = {}

        for name, specification in self.hpo["search_space"].items():
            options = dict(specification)
            kind = options.pop("type")

            if kind == "float":
                value = trial.suggest_float(name, **options)
            elif kind == "int":
                value = trial.suggest_int(name, **options)
            elif kind == "categorical":
                value = trial.suggest_categorical(name, **options)
            else:
                raise ValueError(f"Distribuição desconhecida para {name}: {kind}")

            sampled[name] = value

        return sampled

    def _build_trial_config(self, trial):
        """Build an independent effective configuration for this trial."""
        config = deepcopy(self.config)

        sampled = self._sample_parameters(trial)
        config["training_arguments"].update(sampled)

        study_name = self.hpo["study_name"]
        config["tracking"]["group_name"] = study_name
        config["tracking"]["run_name"] =  f"{study_name}-trial-{trial.number}"

        # Cada trial guarda os seus resultados localmente.
        config["hub"] = dict(config.get("hub") or {})
        config["hub"]["push_to_hub"] = False

        return config

    def _objective(self, trial):
        """Train a fresh experiment and return its validation objective."""
        config = self._build_trial_config(trial)
        experiment = None

        try:
            # O construtor aplica a seed antes de carregar o modelo.
            experiment = TrainingTranslationScript(config=config)
            result = experiment.run()

            trial.set_user_attr("run_dir", result["run_dir"])
            trial.set_user_attr( "best_checkpoint", result["best_checkpoint"])

            metric_name = self.hpo["objective"]
            metrics = result["validation_metrics"]

            if metric_name not in metrics:
                raise ValueError(f"Métrica {metric_name!r} ausente. Métricas disponíveis: {sorted(metrics)}")

            value = float(metrics[metric_name])

            if not math.isfinite(value):
                raise ValueError( f"Valor inválido para {metric_name}: {value}")
            return value

        finally:
            # O run() do treino já fecha o W&B.
            experiment = None
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self):
        """Create or reopen the study and execute sequential trials."""
        study = optuna.create_study(
            study_name=self.hpo["study_name"],
            storage=self.hpo["storage"],
            load_if_exists=True,
            direction=self.hpo["direction"],
            sampler=optuna.samplers.TPESampler(seed=self.hpo["sampler_seed"]),
            pruner=optuna.pruners.NopPruner(),
        )

        study.optimize( self._objective, n_trials=self.hpo["trials"],  n_jobs=1)
        logger.info("Melhor trial: %s", study.best_trial.number)
        logger.info("Melhor resultado: %s", study.best_value)
        logger.info("Melhores parâmetros: %s", study.best_params)

        return study


def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO using the shared translation trainer.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    experiment = HPOTrainingScript(config_path=args.config)
    experiment.run()


if __name__ == "__main__":
    main()
