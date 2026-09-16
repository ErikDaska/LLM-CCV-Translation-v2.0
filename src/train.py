import os

# Redirect HF Cache, W&B, and Temp directories to the 1TB network drive
os.environ["HF_HOME"] = "/home/criolo/storage/.cache/huggingface"
os.environ["WANDB_DIR"] = "/home/criolo/storage/.cache/wandb"
os.environ["TMPDIR"] = "/home/criolo/storage/tmp"

for path in [os.environ["HF_HOME"], os.environ["WANDB_DIR"], os.environ["TMPDIR"]]:
    os.makedirs(path, exist_ok=True)

import logging
import yaml
import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)


class TrainingTranslationScript:
    def __init__(self, config_path: str):
        print("CUDA Available:", torch.cuda.is_available())
        wandb.login(key=os.environ["WANDB_API_KEY"])

        logging.info(f"Config YAML file parsing from {config_path}")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise EnvironmentError("HF_TOKEN is not set in environment or .env file.")

        logging.info(f'Dataset Loading {self.config["dataset"]}')
        self.dataset = load_dataset(self.config["dataset"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"],
            token=self.token,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config["model"],
            token=self.token,
        )

        if self.config.get("gradient_checkpointing", True):
            self.model.config.use_cache = False

        self.max_length = self.config["max_length"]
        self.src_lang = self.config["src_lang"]
        self.tgt_lang = self.config["tgt_lang"]

        # Metrics
        self.sacrebleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")
        self.meteor = evaluate.load("meteor")
        self.ter = evaluate.load("ter")

        # Resolve storage directory safely
        raw_output_dir = self.config.get(
            "output_base_dir",
            "/home/criolo/storage/models_outputs/translation_models",
        )
        self.output_base_dir = os.path.abspath(raw_output_dir)

        if "mbart" in self.config["model"]:
            self.model, self.tokenizer = self._add_mbart_special_token(self.model, self.tokenizer)
        elif "nllb" in self.config["model"]:
            self.tokenizer.src_lang = self.src_lang


    def _add_mbart_special_token(self, model, tokenizer):
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.src_lang]},
            replace_extra_special_tokens=False,
        )

        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))

        token_id = tokenizer.convert_tokens_to_ids(self.src_lang)

        if hasattr(tokenizer, "lang_code_to_id"):
            tokenizer.lang_code_to_id[self.src_lang] = token_id
        if hasattr(self.tokenizer, "id_to_lang_code"):
            tokenizer.id_to_lang_code[token_id] = self.src_lang
        return model, tokenizer

    def preprocess_function(self, examples):
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        return self.tokenizer(
            examples["original_sentence"],
            text_target=examples["translation"],
            max_length=self.max_length,
            truncation=True,
        )

    def compute_metrics(self, eval_preds):
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

    def model_init(self):
        """
        This method was implemented to initialize the model and tokenizer for HPO.
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"],
            token=self.token,
        )

        tokenizer.src_lang = self.src_lang
        tokenizer.tgt_lang = self.tgt_lang

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config["model"],
            token=self.token,
        )

        model, tokenizer = self._add_mbart_special_token(model, tokenizer)

        return model

    def _hp_space(self, trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                5e-6,
                5e-5,
                log=True,
            ),

            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size",
                [8, 16, 32],
            ),

            "num_train_epochs": trial.suggest_int(
                "num_train_epochs",
                3,
                10,
            ),

            "label_smoothing_factor": trial.suggest_float(
                "label_smoothing_factor",
                0.0,
                0.2,
            ),

            "warmup_ratio": trial.suggest_float(
                "warmup_ratio",
                0.0,
                0.15,
            ),

            "weight_decay": trial.suggest_float(
                "weight_decay",
                1e-4,
                0.1,
                log=True,
            ),
        }

    def _hp_name(self, trial):
        group_name = self.config.get("group_name", "hpo")
        return f"{group_name}_trial_{trial.number:02d}"

    def manual_training(self, tokenized_datasets, data_collator):
        logging.info("Initializing W&B...")
        model_name_clean = self.config["model"].split("/")[-1]
        run_name = self.config.get("run_name", f"cpt_{model_name_clean}")

        run = wandb.init(
            dir=os.environ["TMPDIR"],
            project=self.config.get("project", "llm-ccv"),
            group=self.config.get("group_name", None),
            job_type="training",
            name=run_name,
            config={**self.config},
        )

        output_dir = os.path.join(self.output_base_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Model output directory: {output_dir}")

        # Refactor for optimization
        training_kwargs = {
            "output_dir": output_dir,
            "run_name": run_name,
            "num_train_epochs": self.config.get("num_train_epochs", 3),
            "per_device_train_batch_size": self.config.get("per_device_train_batch_size", 64),
            "per_device_eval_batch_size": self.config.get("per_device_eval_batch_size", 64),
            "learning_rate": float(self.config.get("learning_rate", 2e-5)),
            "warmup_steps": self.config.get("warmup_steps", 100),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "logging_steps": self.config.get("logging_steps", 10),
            "eval_strategy": self.config.get("eval_strategy", "epoch"),
            "save_strategy": self.config.get("save_strategy", "epoch"),
            "save_total_limit": self.config.get("save_total_limit", 3),
            "predict_with_generate": True,
            "generation_max_length": self.max_length,
            "generation_num_beams": self.config.get("generation_num_beams", 5),
            "load_best_model_at_end": True,
            "metric_for_best_model": self.config.get("metric_for_best_model", "eval_SacreBleu"),
            "greater_is_better": self.config.get("greater_is_better", True),
            "max_grad_norm": 1.0,
            "logging_nan_inf_filter": True,
            "fp16": self.config.get("fp16", False),
            "bf16": self.config.get("bf16", True),
            "gradient_checkpointing": self.config.get("gradient_checkpointing", True),
            "report_to": "wandb",
        }

        args = Seq2SeqTrainingArguments(**training_kwargs)

        callbacks = []
        patience = self.config.get("early_stopping_patience")
        if patience:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        trainer = Seq2SeqTrainer(
            model=self.model if not self.config.get("hyperparameter_search", False) else None,
            model_init=self._model_init if self.config.get("hyperparameter_search", False) else None,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        trainer.train()

        # This puts back to the default being en_XX, otherwise when loading the model it will give an error
        if self.model.config.model_type == "mbart":
            self.tokenizer.src_lang = "en_XX"

        logging.info("Saving model locally...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if self.config.get("push_to_hub", False):
            hub_repo_id = self.config.get("hub_repo_id", "")
            if not hub_repo_id:
                raise ValueError("push_to_hub is enabled but hub_repo_id is missing in config.")
            hub_private = self.config.get("hub_private", True)
            logging.info(f"Pushing model to HF Hub: {hub_repo_id}")
            self.model.push_to_hub(hub_repo_id, token=self.token, private=hub_private)
            self.tokenizer.push_to_hub(hub_repo_id, token=self.token, private=hub_private)

        if run is not None:
            run.finish()

    def hpo_training(self, tokenized_datasets, data_collator):

        project_name = self.config.get(
            "project",
            "Translation Models",
        )

        group_name = self.config.get(
            "group_name",
            "hpo_experiment",
        )

        hpo_output_dir = os.path.join(
            self.output_base_dir,
            "hpo_trials",
        )

        os.environ["WANDB_PROJECT"] = project_name
        os.environ["WANDB_RUN_GROUP"] = group_name
        os.environ["WANDB_LOG_MODEL"] = "false"

        training_args = Seq2SeqTrainingArguments(
            output_dir=hpo_output_dir,

            report_to=["wandb"],

            eval_strategy="epoch",
            predict_with_generate=True,

            generation_max_length=self.max_length,
            generation_num_beams=self.config.get(
                "generation_num_beams",
                5,
            ),

            logging_strategy="steps",
            logging_steps=self.config.get(
                "logging_steps",
                10,
            ),

            save_strategy="no",

            metric_for_best_model="eval_chrf",
            greater_is_better=True,

            bf16=self.config.get("bf16", True),
            fp16=self.config.get("fp16", False),

            gradient_checkpointing=self.config.get(
                "gradient_checkpointing",
                True,
            ),
        )

        trainer = Seq2SeqTrainer(
            model_init=self.model_init,
            args=training_args,

            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],

            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        best_run = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=self._hp_space,
            hp_name=self._hp_name,
            n_trials=self.config.get("hpo_trials", 10),
            backend="optuna",
        )

        logging.info("Best HPO run: %s", best_run)

        return best_run

    def run(self):
        tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, pad_to_multiple_of=8
        )
        if self.config.get("hyperparameter_search"):
            self.hpo_training(tokenized_datasets, data_collator)
        else:
            (self.manual_training(tokenized_datasets, data_collator))


if __name__ == "__main__":
    script = TrainingTranslationScript(config_path="configs/config.yaml")
    #script = TrainingTranslationScript(config_path="configs/config_hpo.yaml")
    script.run()