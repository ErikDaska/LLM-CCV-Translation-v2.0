"""Train mBART/NLLB translation models or run Optuna searches with local tracking.

Cache directories are configured at import time. Run records start in ``run()``;
initialization failures during authentication or model loading precede tracking.
"""
import os

from dotenv import load_dotenv

load_dotenv()
for variable in ('HF_HOME', 'WANDB_DIR', 'TMPDIR'):
    if os.getenv(variable):
        os.makedirs(os.environ[variable], exist_ok=True)

import logging
from utils.configuration import load_training_configuration
import evaluate
import numpy as np
import torch
import wandb
from utils.run_tracking import RunTracking
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    enable_full_determinism,
    set_seed,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)


class TrainingTranslationScript:
    """Prepare translation resources and dispatch manual training or HPO."""
    def __init__(self, config_path: str):
        """Load YAML settings, authenticate, and initialize data, model and metrics.

        Args:
            config_path: YAML path resolved relative to the working directory.

        Requires WANDB_API_KEY and HF_TOKEN in the environment. Resource loading
        happens here, before the tracked execution starts in ``run()``.
        """
        print("CUDA Available:", torch.cuda.is_available())
        logging.info(f"Config YAML file parsing from {config_path}")
        self.config = load_training_configuration(config_path)

        wandb.login(key=os.environ["WANDB_API_KEY"])
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise EnvironmentError("HF_TOKEN is not set in environment or .env file.")


        logging.info(f'Dataset Loading {self.config["data"]["dataset"]}')
        self.dataset = load_dataset(self.config["data"]["dataset"])

        self.max_length = self.config["data"]["max_length"]
        self.src_lang = self.config["model"]["src_lang"]
        self.tgt_lang = self.config["model"]["tgt_lang"]
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        # Load metrics
        self.sacrebleu, self.chrf, self.meteor, self.ter = self.load_metrics()
        # Setup paths and reproducibility
        self.output_base_dir = self.setup_reproducibility()

    def load_metrics(self):
        # Metrics
        sacrebleu = evaluate.load("sacrebleu")
        chrf = evaluate.load("chrf")
        meteor = evaluate.load("meteor")
        ter = evaluate.load("ter")
        return sacrebleu, chrf, meteor, ter

    def setup_reproducibility(self):
        training = self.config['training_arguments']
        if training['full_determinism']:
            enable_full_determinism(training['seed'])
        else:
            set_seed(training['seed'])

        # Resolve storage directory safely
        raw_output_dir = self.config["tracking"]["output_base_dir"]
        return os.path.abspath(raw_output_dir)


    def _load_model_and_tokenizer(self):
        """Return a fresh model/tokenizer pair prepared for manual training or HPO.

        Raises:
            ValueError: The model/tokenizer family is neither mBART nor NLLB.
        """
        # Reset before loading weights AND initializing any new token embeddings.
        # Every HPO trial starts from the same random initialization conditions.

        model_id = self.config["model"]["model_name"]
        # mBART requires a native language during construction, even when
        # the saved tokenizer contains a custom language token.
        if "mbart" in model_id:
            tokenizer_kwargs = {"src_lang": "en_XX"}
        else:
            tokenizer_kwargs = {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.token, **tokenizer_kwargs)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=self.token)
        return self._prepare_languages(model, tokenizer)

    def _prepare_languages(self, model, tokenizer):
        """Register the source token, select languages and configure generation.

        Mutates and returns the supplied model and tokenizer. Embeddings are
        resized only when tokens are added. The target token must already exist.

        Raises:
            ValueError: The target language token is missing from the vocabulary.
        """
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.src_lang]},
            replace_extra_special_tokens=False,
        )

        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))

        token_id = tokenizer.convert_tokens_to_ids(self.src_lang)

        if hasattr(tokenizer, "lang_code_to_id"):
            tokenizer.lang_code_to_id[self.src_lang] = token_id
        if hasattr(tokenizer, "id_to_lang_code"):
            tokenizer.id_to_lang_code[token_id] = self.src_lang
        target_id = tokenizer.convert_tokens_to_ids(self.tgt_lang)
        if target_id is None or target_id == tokenizer.unk_token_id:
            raise ValueError(f"Unknown target language code: {self.tgt_lang}")

        tokenizer.src_lang = self.src_lang
        tokenizer.tgt_lang = self.tgt_lang
        model.generation_config.forced_bos_token_id = target_id
        if self.config["training_arguments"]["gradient_checkpointing"]:
            model.config.use_cache = False
        return model, tokenizer

    def preprocess_function(self, examples):
        """Tokenize a batch of original_sentence/translation pairs with truncation."""
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        return self.tokenizer(
            examples["original_sentence"],
            text_target=examples["translation"],
            max_length=self.max_length,
            truncation=True,
        )

    def compute_metrics(self, eval_preds):
        """Decode generated token IDs and return BLEU, chrF, METEOR and TER.

        Args:
            eval_preds: Predictions and labels provided by Seq2SeqTrainer.
                Label padding uses -100 and is replaced before decoding.

        Returns:
            Metric scores; higher is better except for TER.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        sacrebleu_result = self.sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = self.chrf.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_results = self.meteor.compute(predictions=decoded_preds, references=decoded_labels)
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

    def manual_training(self, tokenized_datasets, data_collator):
        """Train, evaluate the selected checkpoint, export and optionally publish.

        Requires tracking initialized by ``run()`` and train/validation splits.
        Writes arguments, metrics, history, checkpoint selection and W&B identity
        to the run directory. ``run()`` handles failures and closes W&B.
        """
        logging.info("Initializing W&B...")
        model_name_clean = self.config["model"]["model_name"].split("/")[-1]
        tracking_parameters = self.config["tracking"]
        run_name = tracking_parameters.get("run_name", f"cpt_{model_name_clean}")


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

        output_dir = str(self.tracking.path / 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Model output directory: {output_dir}")

        # Build the effective arguments separately from the requested YAML config.
        training_kwargs = self._configured_training_kwargs()
        training_kwargs.update(output_dir=output_dir, run_name=run_name)

        args = Seq2SeqTrainingArguments(**training_kwargs)
        self.tracking.write('training_args.json', args.to_dict())

        callbacks = []
        # TODO: Rever
        #patience = self.config.get("early_stopping_patience")
        #if patience:
        #    if not args.load_best_model_at_end:
        #        raise ValueError('Early stopping requires load_best_model_at_end: true.')
        #    callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        train_result = trainer.train()
        self.tracking.write('metrics.json', {'train': train_result.metrics})
        self.tracking.write('history.json', trainer.state.log_history)
        self.tracking.write('selection.json', {
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_metric': trainer.state.best_metric,
            'metric_for_best_model': args.metric_for_best_model,
        })
        # The Trainer restored the best checkpoint; evaluate that model explicitly.
        validation_metrics = trainer.evaluate()
        self.tracking.write('metrics.json', {'train': train_result.metrics,
                                            'validation': validation_metrics})

        # This puts back to the default being en_XX, otherwise when loading the model it will give an error
        if self.model.config.model_type == "mbart":
            self.tokenizer.src_lang = "en_XX"

        logging.info("Saving model locally...")
        model_dir = str(self.tracking.path / 'model')
        trainer.save_model(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.chcek_push_to_hub()


    def check_push_to_hub(self):
        hub = self.config["hub"]
        if not hub:
            print("No hub information set in config file. Skipping it...")
        if hub["push_to_hub"]:
            hub_repo_id = hub["hub_repo_id"]
            if not hub_repo_id:
                raise ValueError("push_to_hub is enabled but hub_repo_id is missing in config.")
            hub_private = hub["hub_private"]
            logging.info(f"Pushing model to HF Hub: {hub_repo_id}")
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
            self._run_training()
            self.tracking.status('completed')
            exit_code = 0
        except BaseException as error:
            self.tracking.fail(error)
            callback = getattr(self, 'trial_tracking', None)
            if callback is not None and callback.trial is not None:
                callback.trial.fail(error)
            raise
        finally:
            if wandb.run is not None:
                wandb.finish(exit_code=exit_code)

    def _run_training(self):
        """Tokenize dataset splits, create the collator and dispatch the chosen mode."""
        tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, pad_to_multiple_of=8)
        self.manual_training(tokenized_datasets, data_collator)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train a translation model or run HPO.")
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs/smoke/manual.yaml'))
    script = TrainingTranslationScript(config_path=parser.parse_args().config)
    script.run()
