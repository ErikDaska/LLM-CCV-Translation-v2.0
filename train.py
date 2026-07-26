import evaluate
import numpy as np
import os
import wandb

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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config["model"])

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
        #TODO: Ter special tokens para variantes distintas
        special_tokens = {
            "additional_special_tokens": ["kea_Latn"]
        }

        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        kea_id = self.tokenizer.convert_tokens_to_ids("kea_Latn")

        # Register as a valid language token
        self.tokenizer.lang_code_to_id["kea_Latn"] = kea_id
        self.tokenizer.id_to_lang_code[kea_id] = "kea_Latn"

        if hasattr(self.tokenizer, "fairseq_tokens_to_ids") and hasattr(self.tokenizer, "fairseq_ids_to_tokens"):
            self.tokenizer.fairseq_tokens_to_ids["kea_Latn"] = kea_id
            self.tokenizer.fairseq_ids_to_tokens[kea_id] = "kea_Latn"

        # --------------------------------------------------------
        # NEW:
        # Initialize the new language embedding using Portuguese
        # instead of leaving it randomly initialized.
        # --------------------------------------------------------

        pt_id = self.tokenizer.lang_code_to_id["pt_XX"]
        self.model.model.shared.weight.data[kea_id] = (
            self.model.model.shared.weight.data[pt_id].clone()
        )
        # Generate Portuguese during inference
        self.model.config.forced_bos_token_id = pt_id


    def test_additional_special_tokens(self):
        print("Special token id added: ", self.tokenizer.lang_code_to_id["kea_Latn"])

        self.tokenizer.src_lang = "kea_Latn"

        print("Inpud ids: ", self.tokenizer("N ta bai kasa"))

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
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        args = Seq2SeqTrainingArguments(
            output_dir="test",
            eval_strategy="steps",
            save_strategy="steps",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
            fp16=True,
            logging_strategy="steps",
            logging_steps=100,
            report_to="wandb",
            warmup_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model = "bleu",
            greater_is_better = True,
            push_to_hub=True,
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

        trainer.save_model("mbart-kea")
        self.tokenizer.save_pretrained("mbart-kea")


if __name__ == "__main__":
    path = "config.yaml"
    trainer_Script = TrainingTranslationScript(config_path=path)
    trainer_Script._fn_add_special_tokens()
    trainer_Script.test_additional_special_tokens()
    trainer_Script.run()