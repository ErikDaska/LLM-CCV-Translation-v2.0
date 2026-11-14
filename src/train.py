"""Train mBART/NLLB translation models

Cache directories are configured at import time. Run records start in ``run()``;
initialization failures during authentication or model loading precede tracking.
"""
import os
from dotenv import load_dotenv

load_dotenv()
for variable in ('HF_HOME', 'WANDB_DIR', 'TMPDIR'):
    if os.getenv(variable):
        os.makedirs(os.environ[variable], exist_ok=True)

from functools import partial
import logging
from utils.configuration import validate_training_config
import evaluate
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from utils.run_tracking import RunTracking
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    enable_full_determinism,
    set_seed,
    AutoConfig,
    M2M100Tokenizer,
)

logging.basicConfig(level=logging.INFO)


class TrainingRunExtension:
    """Optional hooks for run-specific behaviour; every method is a no-op.

    Extensions receive lifecycle events so the training script stays free of
    domain concerns (experiment controllers, trackers, resumable studies).
    """

    def build_callbacks(self):
        """Return extra callbacks to register on the Trainer."""
        return []

    def on_run_started(self, wandb_run):
        """Called after the W&B run is open and its identity recorded."""

    def on_completed(self, validation_metrics):
        """Called after final evaluation and before the model export."""

    def on_error(self, error):
        """Return a status name for an expected outcome, or None for a failure."""
        return None


class TrainingTranslationScript:
    """Prepare translation resources and run manual training."""

    def __init__(self, config, *, extension=None):
        """Initialize data, model and metrics from a training configuration.

        Args:
            config: Settings mapping validated by ``validate_training_config``.
            extension: Optional ``TrainingRunExtension`` with run-specific
                callbacks and lifecycle hooks; defaults to a no-op.

        Requires WANDB_API_KEY and HF_TOKEN in the environment. Resource loading
        happens here, before the tracked execution starts in ``run()``.
        """
        logging.info("CUDA Available: %s", torch.cuda.is_available())
        validate_training_config(config)
        self.config = config
        self.extension = extension if extension is not None else TrainingRunExtension()

        self._setup_reproducibility()
        self._authenticate()
        self._load_dataset()
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self._load_metrics()

    def _setup_reproducibility(self):
        """Seed the process and resolve the run storage base directory."""
        training = self.config['training_arguments']
        if training['full_determinism']:
            enable_full_determinism(training['seed'])
        else:
            set_seed(training['seed'])

        raw_output_dir = self.config["tracking"]["output_base_dir"]
        self.output_base_dir = os.path.abspath(raw_output_dir)

    def _authenticate(self):
        """Log in to W&B and require an HF token before any network call."""
        wandb.login(key=os.environ["WANDB_API_KEY"])
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise EnvironmentError("HF_TOKEN is not set in environment or .env file.")

    def _load_dataset(self):
        """Load the parallel corpus and cache the language/max length settings."""
        logging.info('Dataset Loading %s', self.config["data"]["dataset"])
        self.dataset = load_dataset(self.config["data"]["dataset"], token=self.token)

        self.max_length = self.config["data"]["max_length"]
        self.src_lang = self.config["model"]["src_lang"]
        self.tgt_lang = self.config["model"]["tgt_lang"]

    def _load_model_and_tokenizer(self):
        model_id = self.config["model"]["model_name"]
        model_config = AutoConfig.from_pretrained(model_id, token=self.token)
        tokenizer_kwargs = {}

        if model_config.model_type == "mbart":
            tokenizer_kwargs["src_lang"] = "en_XX"
        elif model_config.model_type == "m2m_100":
            tokenizer_kwargs["src_lang"] = "en"

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.token, **tokenizer_kwargs)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=self.token)
        return self._prepare_languages(model, tokenizer)

    def _prepare_languages(self, model, tokenizer):
        is_m2m100 = isinstance(tokenizer, M2M100Tokenizer)

        # Preserva o token nativo quando a origem já é uma língua suportada.
        if is_m2m100 and self.src_lang in tokenizer.lang_code_to_token:
            source_token = tokenizer.lang_code_to_token[self.src_lang]
        else:
            source_token = self.src_lang

        tokenizer.add_special_tokens(
            {"additional_special_tokens": [source_token]},
            replace_extra_special_tokens=False,
        )

        source_id = tokenizer.convert_tokens_to_ids(source_token)

        if source_id is None or source_id == tokenizer.unk_token_id:
            raise ValueError(f"Token de origem inválido: {source_token}")

        if is_m2m100:
            # Regista também línguas personalizadas.
            tokenizer.lang_code_to_token[self.src_lang] = source_token
            tokenizer.lang_code_to_id[self.src_lang] = source_id
            tokenizer.lang_token_to_id[source_token] = source_id
            tokenizer.id_to_lang_token[source_id] = source_token

            if self.tgt_lang not in tokenizer.lang_code_to_id:
                raise ValueError( f"Língua de destino desconhecida: {self.tgt_lang}")

            target_id = tokenizer.get_lang_id(self.tgt_lang)
        else:
            target_id = tokenizer.convert_tokens_to_ids(self.tgt_lang)

        if target_id is None or target_id == tokenizer.unk_token_id:
            raise ValueError(f"Língua de destino desconhecida: {self.tgt_lang}")

        # Garante espaço para os IDs sem reduzir embeddings já existentes.
        required_size = max( len(tokenizer), source_id + 1, target_id + 1)
        current_size = model.get_input_embeddings().num_embeddings

        if required_size > current_size:
            model.resize_token_embeddings(required_size)

        tokenizer.src_lang = self.src_lang
        tokenizer.tgt_lang = self.tgt_lang

        model.generation_config.forced_bos_token_id = target_id

        if self.config["training_arguments"]["gradient_checkpointing"]:
            model.config.use_cache = False

        return model, tokenizer

    def _load_metrics(self):
        self.sacrebleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")
        self.meteor = evaluate.load("meteor")
        self.ter = evaluate.load("ter")

    def preprocess_function(self, examples):
        """Tokenize a batch of original_sentence/translation pairs with truncation."""
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        return self.tokenizer(
            examples["original_sentence"],
            text_target=examples["translation"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

    def compute_metrics(self, eval_preds, *, references):
        """Decode generated token IDs and return BLEU, chrF, METEOR and TER.

            Args:
                eval_preds: Predictions and labels provided by Seq2SeqTrainer.
                    Label padding uses -100 and is replaced before decoding.

            Returns:
                Metric scores; higher is better except for TER.
        """
        preds = eval_preds.predictions

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where( preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode( preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        # Referências originais, sem tokenização nem truncagem.
        decoded_labels = [
            [ref.strip() for ref in item] if isinstance(item, (tuple, list))
            else [item.strip()]
            for item in references
        ]

        if len(decoded_preds) != len(decoded_labels):
            raise ValueError(f"Número de previsões ({len(decoded_preds)}) diferente do número de referências ({len(decoded_labels)}).")

        # A partir daqui, mantém as chamadas às métricas e o return atuais.
        sacrebleu_result = self.sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = self.chrf.compute(predictions=decoded_preds, references=decoded_labels)
        # METEOR is single-reference; report it on the first reference only.
        meteor_results = self.meteor.compute(
            predictions=decoded_preds,
            references=[refs[0] for refs in decoded_labels],
        )
        ter_result = self.ter.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "SacreBleu": sacrebleu_result["score"],
            "chrf": chrf_result["score"],
            "meteor": meteor_results["meteor"],
            "ter": ter_result["score"],
        }

    def _configured_training_kwargs(self):
        """Return the single YAML source of effective Trainer settings."""
        return dict(self.config['training_arguments'])

    def _resolve_run_name(self):
        """Derive the W&B run name, falling back to a model-based default."""
        model_name_clean = self.config["model"]["model_name"].split("/")[-1]
        tracking_parameters = self.config["tracking"]
        return tracking_parameters.get("run_name", f"cpt_{model_name_clean}")

    def _start_wandb_run(self, run_name):
        """Open the tracked W&B run and record its identity on disk."""
        logging.info("Initializing W&B...")
        tracking_parameters = self.config["tracking"]
        run = wandb.init(
            id=self.tracking.run_id,
            dir=str(self.tracking.path),
            project=tracking_parameters.get("project", "llm-ccv-translation"),
            group=tracking_parameters.get("group_name", None),
            job_type="training",
            name=run_name,
            config=self.config,
        )
        self.tracking.write('wandb.json', {'id': run.id, 'url': run.url})
        self.extension.on_run_started(run)
        return run

    def _build_training_arguments(self, output_dir, run_name):
        """Build Seq2SeqTrainingArguments from the YAML settings and persist them."""
        training_kwargs = self._configured_training_kwargs()
        training_kwargs.update(output_dir=output_dir, run_name=run_name)

        args = Seq2SeqTrainingArguments(**training_kwargs)
        self.tracking.write('training_args.json', args.to_dict())
        return args

    def _build_callbacks(self, args):
        """Assemble early stopping plus any extension-provided callbacks."""
        callbacks = []
        patience = self.config.get("early_stopping_patience")
        if patience:
            if not args.load_best_model_at_end:
                raise ValueError('Early stopping requires load_best_model_at_end: true.')
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        callbacks.extend(self.extension.build_callbacks())
        return callbacks

    def _build_validation_view(self, tokenized_validation):
        """Return the eval split and its references (optionally multi-reference)."""
        validation_dataset = tokenized_validation
        if self.config["data"].get("multi_reference_validation", False):
            from utils.validation_refs import build_multireference_view

            positions, references = build_multireference_view(self.dataset["validation"])
            validation_dataset = validation_dataset.select(positions)
            logging.info("Validation: %d pairs grouped into %d unique sources",
                         len(self.dataset["validation"]), len(positions))
        else:
            references = tuple(self.dataset["validation"]["translation"])
        return validation_dataset, references

    def _build_trainer(self, tokenized_datasets, validation_dataset, references,
                       data_collator, args, callbacks):
        """Create the Seq2SeqTrainer bound to the validation metrics."""
        validation_compute_metrics = partial(self.compute_metrics, references=references)
        return Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=validation_compute_metrics,
            callbacks=callbacks,
        )

    def _train_and_evaluate(self, trainer):
        """Run the training loop, then evaluate the restored best checkpoint."""
        train_result = trainer.train()
        # The Trainer restored the best checkpoint; evaluate that model explicitly.
        validation_metrics = trainer.evaluate()
        return train_result, validation_metrics

    def _record_results(self, trainer, args, train_result, validation_metrics):
        """Write metrics, history and checkpoint selection to the run directory."""
        self.tracking.write('metrics.json', {'train': train_result.metrics,
                                             'validation': validation_metrics})
        self.tracking.write('history.json', trainer.state.log_history)
        self.tracking.write('selection.json', {
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_metric': trainer.state.best_metric,
            'metric_for_best_model': args.metric_for_best_model,
        })
        self.extension.on_completed(validation_metrics)

    def _export_final_model(self, trainer):
        """Reset mBART source language, save locally and optionally publish."""
        # This puts back to the default being en_XX, otherwise when loading the model it will give an error
        if self.model.config.model_type == "mbart":
            self.tokenizer.src_lang = "en_XX"

        if self.config.get("save_final_model", True):
            logging.info("Saving final model locally...")
            model_dir = str(self.tracking.path / 'model')
            trainer.save_model(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            self.check_push_to_hub()
        else:
            logging.info("Skipping final model export (save_final_model is false)")

    def manual_training(self, tokenized_datasets, data_collator):
        """Train, evaluate the selected checkpoint, export and optionally publish.

        Requires tracking initialized by ``run()`` and train/validation splits.
        Writes arguments, metrics, history, checkpoint selection and W&B identity
        to the run directory. ``run()`` handles failures and closes W&B.
        """
        run_name = self._resolve_run_name()
        self._start_wandb_run(run_name)

        output_dir = str(self.tracking.path / 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Model output directory: %s", output_dir)

        args = self._build_training_arguments(output_dir, run_name)
        callbacks = self._build_callbacks(args)
        validation_dataset, references = self._build_validation_view(
            tokenized_datasets["validation"])
        trainer = self._build_trainer(tokenized_datasets, validation_dataset, references,
                                      data_collator, args, callbacks)

        train_result, validation_metrics = self._train_and_evaluate(trainer)
        self._record_results(trainer, args, train_result, validation_metrics)
        self._export_final_model(trainer)

        return {
            "validation_metrics": validation_metrics,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "run_dir": str(self.tracking.path),
        }

    def check_push_to_hub(self):
        """Upload model and tokenizer when the config enables the Hub push."""
        hub = self.config.get("hub") or {}
        if not hub.get("push_to_hub", False):
            logging.info("Skipping Hugging Face Hub upload")
            return
        hub_repo_id = hub.get("hub_repo_id")
        if not hub_repo_id:
            raise ValueError("push_to_hub is enabled but hub_repo_id is missing in config.")
        hub_private = hub.get("hub_private", True)
        logging.info("Pushing model to HF Hub: %s", hub_repo_id)
        self.model.push_to_hub(hub_repo_id, token=self.token, private=hub_private)
        self.tokenizer.push_to_hub(hub_repo_id, token=self.token, private=hub_private)

    def run(self):
        """Track preprocessing and training, record failures and always close W&B.

        Exceptions are recorded and re-raised. Abrupt process termination may
        leave the status as running because cleanup cannot execute in that case.
        """

        self.tracking = RunTracking(self.output_base_dir)
        logging.info('Run %s: %s', self.tracking.run_id, self.tracking.path)
        exit_code = 1
        try:
            self.tracking.write('config.json', self.config)
            result = self._run_training()

            self.tracking.status("completed")
            exit_code = 0
            return result
        except BaseException as error:
            status = self.extension.on_error(error) or "failed"
            if status == "failed":
                self.tracking.fail(error)
            else:
                # Expected outcome declared by the extension, not a failed run.
                self.tracking.status(status)
                exit_code = 0
            raise
        finally:
            if wandb.run is not None:
                wandb.finish(exit_code=exit_code)

    def _run_training(self):
        """Tokenize dataset splits, create the collator and run training."""
        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset["train"].column_names)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, pad_to_multiple_of=8)
        return self.manual_training(tokenized_datasets, data_collator)

@hydra.main(config_path="configs/training", config_name="smoke_manual", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose the Hydra config and run one training run.

    Any ``key=value`` argument overrides the YAML (e.g.
    ``training_arguments.learning_rate=3e-5``) and ``-m`` sweeps combinations;
    each run stores the resolved config under ``outputs/.../.hydra/config.yaml``.
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    TrainingTranslationScript(config=config).run()


if __name__ == "__main__":
    main()
