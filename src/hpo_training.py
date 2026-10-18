"""Resumable, sequential overnight Optuna HPO for the shared seq2seq trainer."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import gc
import json
import logging
import math
import shutil

# train configures HF_HOME / WANDB_DIR / TMPDIR before importing heavy libraries.
from train import TrainingTranslationScript
from utils.configuration import load_training_configuration

import optuna
import torch

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "smoke"/ "hpo.yaml"
logger = logging.getLogger(__name__)


class RecoverableTrialError(RuntimeError):
    """GPU out-of-memory: mark this trial failed and continue the study."""


class HPOTrainingScript:
    def __init__(self, config_path=CONFIG_PATH):
        self.config = load_training_configuration(str(config_path))
        self.hpo = self.config["hpo"]
        self._validate()

    def _validate(self):
        for key in ("study_name", "storage", "objective", "direction", "sampler_seed", "trials", "search_space"):
            if key not in self.hpo:
                raise ValueError(f"Missing hpo.{key} in YAML")
        if not self.hpo["search_space"]:
            raise ValueError("hpo.search_space is empty")
        if self.hpo["direction"] not in ("maximize", "minimize"):
            raise ValueError("hpo.direction must be maximize or minimize")
        if self.config["training_arguments"].get("eval_strategy") != "epoch":
            raise ValueError("Pruning callback requires training_arguments.eval_strategy='epoch'")
        if not self.config["training_arguments"].get("predict_with_generate", False):
            raise ValueError("HPO on chrF requires predict_with_generate: true")
        if self.hpo.get("require_dataset_revision", True):
            revision = self.config["data"].get("revision")
            if not revision or "REPLACE" in str(revision):
                raise ValueError("Specify a real data.revision (Hugging Face commit SHA) before HPO")
        if self.config["training_arguments"].get("bf16", False):
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("bf16: true requires a CUDA GPU with BF16 support; set bf16: false and consider fp16")
        if self.config["training_arguments"].get("fp16") and self.config["training_arguments"].get("bf16"):
            raise ValueError("Set either fp16 or bf16, not both")

    def _sample_parameters(self, trial):
        sampled = {}
        for name, spec in self.hpo["search_space"].items():
            options = dict(spec)
            kind = options.pop("type")
            if kind == "float":
                value = trial.suggest_float(name, **options)
            elif kind == "int":
                value = trial.suggest_int(name, **options)
            elif kind == "categorical":
                value = trial.suggest_categorical(name, **options)
            else:
                raise ValueError(f"Unknown distribution {kind} for {name}")
            sampled[name] = value
        return sampled

    def _build_trial_config(self, trial):
        config = deepcopy(self.config)
        config["training_arguments"].update(self._sample_parameters(trial))
        # Optional constant *effective* train batch size for single-GPU studies.
        effective = self.hpo.get("effective_batch_size")
        if effective is not None:
            per_device = int(config["training_arguments"].get("per_device_train_batch_size", 8))
            if effective % per_device:
                raise ValueError("effective_batch_size must be divisible by per_device_train_batch_size")
            config["training_arguments"]["gradient_accumulation_steps"] = effective // per_device

        study_name = self.hpo["study_name"]
        config["tracking"]["group_name"] = study_name
        config["tracking"]["run_name"] = f"{study_name}-trial-{trial.number:03d}"
        config["hub"] = dict(config.get("hub") or {})
        config["hub"]["push_to_hub"] = False
        # HPO is for selecting hyperparameters; retrain the winning configuration later.
        config["save_final_model"] = False
        return config

    def _objective(self, trial):
        config = self._build_trial_config(trial)
        experiment = None
        run_dir = None
        try:
            experiment = TrainingTranslationScript(config=config, optuna_trial=trial)
            result = experiment.run()
            run_dir = Path(result["run_dir"])
            metric_name = self.hpo["objective"]
            score = result["validation_metrics"].get(metric_name)
            if score is None:
                raise ValueError(f"Metric {metric_name!r} absent from validation: {sorted(result['validation_metrics'])}")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError(f"Metric {metric_name!r} is not finite: {score}")

            trial.set_user_attr("run_dir", str(run_dir))
            trial.set_user_attr("validation_metric", score)
            trial.set_user_attr("best_checkpoint_during_trial", str(result["best_checkpoint"]))
            trial.set_user_attr("checkpoints_retained", bool(self.hpo.get("keep_checkpoints", False)))
            logger.info("Trial %d COMPLETE: %s=%.4f", trial.number, metric_name, score)
            return score

        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                trial.set_user_attr("failure_reason", "cuda_oom")
                logger.exception("Trial %d ran out of memory; continuing after cleanup", trial.number)
                raise RecoverableTrialError(f"CUDA OOM in trial {trial.number}") from error
            raise
        finally:
            # Preserve JSON summaries and W&B logs but avoid accumulating large checkpoints.
            if experiment is not None:
                tracking = getattr(experiment, "tracking", None)
                if run_dir is None and tracking is not None:
                    run_dir = getattr(tracking, "path", None)
                # Drop model reference before the next independent trial.
                del experiment
            if run_dir is not None and not self.hpo.get("keep_checkpoints", False):
                checkpoint_dir = Path(run_dir) / "checkpoints"
                if checkpoint_dir.is_dir():
                    shutil.rmtree(checkpoint_dir)
                    logger.info("Deleted checkpoints: %s", checkpoint_dir)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _config_fingerprint(self):
        # Prevent silently mixing incompatible configurations in one resumed study.
        identity = {
            "model": self.config["model"],
            "data": self.config["data"],
            "training_arguments": self.config["training_arguments"],
            "early_stopping_patience": self.config.get("early_stopping_patience"),
            "search_space": self.hpo["search_space"],
            "objective": self.hpo["objective"],
            "direction": self.hpo["direction"],
            "effective_batch_size": self.hpo.get("effective_batch_size"),
        }
        return sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()

    def run(self):
        output = Path(self.config["tracking"]["output_base_dir"])
        output.mkdir(parents=True, exist_ok=True)
        min_free = float(self.hpo.get("min_free_disk_gb", 0))
        free_gb = shutil.disk_usage(output).free / (1024 ** 3)
        if free_gb < min_free:
            raise RuntimeError(f"Not enough free space for HPO: {free_gb:.1f} GB < {min_free:.1f} GB")
        logger.info("Disk space available: %.1f GB", free_gb)

        storage_url = self.hpo["storage"]
        if storage_url.startswith("sqlite:///"):
            db_path = storage_url[len("sqlite:///"):]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=int(self.hpo.get("heartbeat_seconds", 60)),
            grace_period=int(self.hpo.get("heartbeat_grace_seconds", 300)),
            engine_kwargs={"connect_args": {"timeout": 60}} if storage_url.startswith("sqlite:") else None,
        )
        study = optuna.create_study(
            study_name=self.hpo["study_name"],
            storage=storage,
            load_if_exists=True,
            direction=self.hpo["direction"],
            sampler=optuna.samplers.TPESampler(
                seed=int(self.hpo["sampler_seed"]),
                n_startup_trials=int(self.hpo.get("n_startup_trials", 5)),
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=int(self.hpo.get("n_startup_trials", 5)),
                n_warmup_steps=int(self.hpo.get("pruning_warmup_epochs", 2)),
                interval_steps=1,
            ),
        )
        # Mark heartbeat-stale RUNNING trials from previous crashed processes as FAIL.
        optuna.storages.fail_stale_trials(study)
        fingerprint = self._config_fingerprint()
        current = study.user_attrs.get("hpo_config_fingerprint")
        if current is not None and current != fingerprint:
            raise ValueError("Existing study uses different model/data/search settings. Choose a new study_name/storage.")
        if current is None:
            study.set_user_attr("hpo_config_fingerprint", fingerprint)

        # Here 'trials' means total attempts across restarts, NOT each call to optimize.
        total = int(self.hpo["trials"])
        remaining = max(0, total - len(study.trials))
        logger.info("Study %s: %d existing, %d new trial slots, timeout=%s seconds",
                    study.study_name, len(study.trials), remaining, self.hpo.get("timeout_seconds"))
        if remaining:
            study.optimize(
                self._objective,
                n_trials=remaining,
                timeout=self.hpo.get("timeout_seconds"),
                n_jobs=1,
                catch=(RecoverableTrialError,),
                gc_after_trial=True,
            )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info("Study complete state counts: %s", {s.name: sum(t.state == s for t in study.trials)
                                                     for s in optuna.trial.TrialState})
        if completed:
            logger.info("Best trial: %s; score: %s; parameters: %s",study.best_trial.number, study.best_value, study.best_params)
        else:
            logger.warning("No completed trials yet; check logs and storage")
        return study


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    HPOTrainingScript(CONFIG_PATH).run()


if __name__ == "__main__":
    main()