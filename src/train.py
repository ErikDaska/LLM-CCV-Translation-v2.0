import evaluate
import numpy as np
import os
import wandb
import tempfile

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
    )
from dotenv import load_dotenv
import yaml
import torch
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)

class TrainingTranslationScript:
    def __init__(self, config_path: str):

        print(torch.cuda.is_available())
        wandb.login(key=os.environ["WANDB_API_KEY"])

        logging.info(f'Config YAML file parsing from {config_path}')
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise EnvironmentError("HF_TOKEN is not set. Add it to your .env file or environment.")

        # Dataset Loading
        logging.info(f'Dataset Loading {self.config["dataset"]}')
        self.dataset = load_dataset(self.config["dataset"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config["model"]
        )

        if self.config.get("gradient_checkpointing", True):
            self.model.config.use_cache = False

        self.max_length = self.config["max_length"]
        self.src_lang = self.config["src_lang"]
        self.tgt_lang = self.config["tgt_lang"]
        self.metric = evaluate.load("sacrebleu")


    def _fn_add_special_tokens(self):
        """
        Void fn
        :param tokenizer:
        :param model:
        :return:
        """
        special_tokens = {
            "additional_special_tokens": [self.src_lang]
        }

        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        kea_id = self.tokenizer.convert_tokens_to_ids(self.src_lang)

        # Register as a valid language token
        self.tokenizer.lang_code_to_id[self.src_lang] = kea_id
        self.tokenizer.id_to_lang_code[kea_id] = self.src_lang

        if hasattr(self.tokenizer, "fairseq_tokens_to_ids") and hasattr(self.tokenizer, "fairseq_ids_to_tokens"):
            self.tokenizer.fairseq_tokens_to_ids[self.src_lang] = kea_id
            self.tokenizer.fairseq_ids_to_tokens[kea_id] = self.src_lang

        # --------------------------------------------------------
        # Initialize the new language embedding using Portuguese
        # --------------------------------------------------------

        pt_id = self.tokenizer.lang_code_to_id[self.tgt_lang]

        embedding = self.model.model.shared.weight.data[pt_id].clone()

        self.model.model.shared.weight.data[kea_id] = embedding

        # Explicitly configure generation
        self.model.config.decoder_start_token_id = pt_id
        self.model.config.forced_bos_token_id = pt_id


    def test_additional_special_tokens(self):
        logging.info(
            f"Special token id added: {self.tokenizer.lang_code_to_id[self.src_lang]}"
        )
        self.tokenizer.src_lang = self.src_lang
        logging.info(self.tokenizer("N ta bai kasa"))

    def preprocess_function(self, examples):
        """
            Tokenize the source and target sentences using the
            correct source and target language codes.
        """
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        model_inputs = self.tokenizer(
            examples["original_sentence"],
            text_target=examples["translation"],
            max_length=self.max_length,
            truncation=True,
        )

        return model_inputs


    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}


    def run(self):
        # Applying the tokenization
        tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )
        print(tokenized_datasets)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, pad_to_multiple_of=8)

        logging.info("Initializing W&B...")
        model_name_clean = self.config["model"].split("/")[-1]
        run_name = self.config.get("run_name", f"cpt_{model_name_clean}")
        group_name = self.config.get("group_name", None)
        run = wandb.init(
            dir=tempfile.gettempdir(),
            project=self.config.get("project", "llm-ccv"),
            group=group_name,
            job_type="training",
            name=run_name,
            config={**self.config},
        )


        output_dir = os.path.join("models_outputs", run_name)
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Setting training arguments...")
        training_kwargs = {
            "output_dir": output_dir,
            "run_name": run_name,
            "num_train_epochs": self.config.get("num_train_epochs", 3),
            "per_device_train_batch_size": self.config.get("per_device_train_batch_size", 4),
            "per_device_eval_batch_size": self.config.get("per_device_eval_batch_size", 4),
            "learning_rate": self.config.get("learning_rate", 2e-5),
            "warmup_steps": self.config.get("warmup_steps", 100),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "logging_steps": self.config.get("logging_steps", 10),
            "eval_strategy": self.config.get("eval_strategy", "steps"),
            "eval_steps": self.config.get("eval_steps", 500),
            "save_strategy": self.config.get("save_strategy", "steps"),
            "save_steps": self.config.get("save_steps", 500),
            "save_total_limit": self.config.get("save_total_limit", 3),
            "predict_with_generate": True,
            "generation_max_length": self.max_length,
            "generation_num_beams": self.config.get("generation_num_beams", 5),
            "load_best_model_at_end": True,
            "metric_for_best_model": "bleu",
            "greater_is_better": True,
            "max_grad_norm": 1.0,
            "logging_nan_inf_filter": True,
            "fp16": self.config.get("fp16", False),
            "bf16": self.config.get("bf16", True),
            "gradient_checkpointing": self.config.get("gradient_checkpointing", True),
            "report_to": "wandb",
        }

        args = Seq2SeqTrainingArguments(
            **training_kwargs
        )

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        logging.info("Saving model locally...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        wandb.finish()

if __name__ == "__main__":
    path = "configs/config.yaml"
    trainer_Script = TrainingTranslationScript(config_path=path)
    trainer_Script._fn_add_special_tokens()
    trainer_Script.test_additional_special_tokens()
    trainer_Script.run()