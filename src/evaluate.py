"""Evaluate a published KEA <-> PT translation model and track results in W&B.

Models/tokenizers and evaluation data are loaded by Hugging Face repository ID.
Only evaluation settings are read from configs/evaluation.yaml; no training
checkpoint directories, local run paths, or command-line arguments are used.

Run: python evaluate_translation.py

Requires: transformers, datasets, huggingface_hub, torch, sacrebleu,
          wandb, python-dotenv, pyyaml, sentencepiece (for some tokenizers).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Respect the same optional cache environment variables as train.py.
load_dotenv()
for variable in ("HF_HOME", "WANDB_DIR", "TMPDIR"):
    if os.getenv(variable):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)

import torch
import wandb
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi
from sacrebleu.metrics import BLEU, CHRF, TER
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOG = logging.getLogger("translation_evaluation")
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "evaluation" / "baseline_nllb.yaml"

def load_evaluation_configuration(path: Path) -> dict[str, Any]:
    """Load a small evaluation-only YAML, in the style of train.py."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The evaluation YAML must be a mapping")

    required = {
        "model": ("model_id", "src_lang", "tgt_lang"),
        "data": ("dataset_id", "split", "source_column", "target_column"),
        "generation": ("batch_size", "max_input_length", "max_new_tokens", "num_beams"),
        "tracking": ("project",),
    }
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing YAML section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing YAML field: {section}.{key}")

    for section, keys in {
        "model": ("model_id", "src_lang", "tgt_lang"),
        "data": ("dataset_id", "split", "source_column", "target_column"),
        "tracking": ("project",),
    }.items():
        for key in keys:
            value = config[section][key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty string")

    for section in ("model", "data"):
        revision = config[section].get("revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError(f"{section}.revision must be null or a revision string")

    for key in required["generation"]:
        value = config["generation"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"generation.{key} must be a positive integer")

    if config["data"]["source_column"] == config["data"]["target_column"]:
        raise ValueError("Source and target columns must differ")
    if config["data"]["split"] != "test":
        raise ValueError("Final evaluation must use data.split: test")

    for key in ("entity", "group", "run_name"):
        value = config["tracking"].get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"tracking.{key} must be null or a string")

    max_rows = config["tracking"].get("preview_rows", 100)
    if type(max_rows) is not int or max_rows < 0:
        raise ValueError("tracking.preview_rows must be a nonnegative integer")
    if type(config["tracking"].get("log_predictions_artifact", True)) is not bool:
        raise ValueError("tracking.log_predictions_artifact must be boolean")
    return config


def normalized_key(text: str) -> str:
    """Group text variants in the same way as the corpus cleaning script."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def group_test_references(
    dataset, source_column: str, target_column: str
) -> tuple[list[dict[str, Any]], int]:
    """One prediction per distinct source, potentially N target references."""
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    invalid_rows = 0

    for position, row in enumerate(dataset):
        source, reference = row.get(source_column), row.get(target_column)
        if not isinstance(source, str) or not isinstance(reference, str):
            invalid_rows += 1
            continue
        source, reference = source.strip(), reference.strip()
        if not source or not reference:
            invalid_rows += 1
            continue

        key = normalized_key(source)
        ref_key = normalized_key(reference)
        if not key or not ref_key:
            invalid_rows += 1
            continue

        if key not in grouped:
            grouped[key] = {
                "source": source,
                "references": [],
                "row_indices": [],
                "_reference_keys": set(),
            }
        record = grouped[key]
        record["row_indices"].append(position)
        if ref_key not in record["_reference_keys"]:
            record["_reference_keys"].add(ref_key)
            record["references"].append(reference)

    items = list(grouped.values())
    for item in items:
        del item["_reference_keys"]
    return items, invalid_rows


def _get_token_id(tokenizer, token: str) -> int:
    """Return a published language-token ID; never create tokens at evaluation."""
    if hasattr(tokenizer, "lang_code_to_id") and token in tokenizer.lang_code_to_id:
        token_id = tokenizer.lang_code_to_id[token]
    else:
        token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Unknown language token {token!r}. Set src_lang/tgt_lang to exactly "
            "the codes used during training and publish the fine-tuned tokenizer "
            "together with the model."
        )
    return int(token_id)


def configure_languages(model, tokenizer, src_lang: str, tgt_lang: str) -> int:
    """Restore native/custom language identifiers and the target BOS token."""
    model_type = model.config.model_type
    # NLLB models also have config.model_type == "m2m_100", but their
    # tokenizers are NllbTokenizer[Fast] and do NOT implement M2M100 maps.
    is_m2m_tokenizer = tokenizer.__class__.__name__.startswith("M2M100Tokenizer")

    if is_m2m_tokenizer:
        if not hasattr(tokenizer, "lang_code_to_id"):
            raise ValueError("Expected an M2M100 tokenizer language mapping")
        for lang in (src_lang, tgt_lang):
            if lang not in tokenizer.lang_code_to_id:
                token_id = _get_token_id(tokenizer, lang)
                tokenizer.lang_code_to_id[lang] = token_id
                if hasattr(tokenizer, "lang_code_to_token"):
                    tokenizer.lang_code_to_token[lang] = lang
                if hasattr(tokenizer, "lang_token_to_id"):
                    tokenizer.lang_token_to_id[lang] = token_id
                if hasattr(tokenizer, "id_to_lang_token"):
                    tokenizer.id_to_lang_token[token_id] = lang
        target_id = int(tokenizer.get_lang_id(tgt_lang))
    else:
        for lang in (src_lang, tgt_lang):
            token_id = _get_token_id(tokenizer, lang)
            # Some mBART tokenizers don't reconstruct custom language mappings.
            if model_type == "mbart" and hasattr(tokenizer, "lang_code_to_id"):
                tokenizer.lang_code_to_id.setdefault(lang, token_id)
                if hasattr(tokenizer, "id_to_lang_code"):
                    tokenizer.id_to_lang_code[token_id] = lang
        target_id = _get_token_id(tokenizer, tgt_lang)

    try:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Unable to configure tokenizer for {src_lang} -> {tgt_lang}. "
            "Check the actual tokenizer class and language IDs in the Hub repo."
        ) from exc

    source_id = _get_token_id(tokenizer, src_lang)
    vocab_size = model.get_input_embeddings().num_embeddings
    if max(source_id, target_id) >= vocab_size:
        raise ValueError(
            "Language token ID is larger than the model's embedding matrix. "
            "The published tokenizer/model are incompatible (perhaps token "
            "embeddings were not saved after adding a custom language)."
        )
    model.generation_config.forced_bos_token_id = target_id
    LOG.info("Direction: %s -> %s; forced_bos_token_id=%d", src_lang, tgt_lang, target_id)
    return target_id


def generate_translations(model, tokenizer, items: list[dict], generation: dict, target_id: int) -> list[str]:
    """Deterministic beam-search translations, one per distinct test source."""
    device = next(model.parameters()).device
    predictions: list[str] = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(items), generation["batch_size"]):
            batch = items[start : start + generation["batch_size"]]
            encodings = tokenizer(
                [item["source"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=generation["max_input_length"],
            ).to(device)
            output = model.generate(
                **encodings,
                forced_bos_token_id=target_id,
                max_new_tokens=generation["max_new_tokens"],
                num_beams=generation["num_beams"],
                do_sample=False,
            )
            predictions.extend(text.strip() for text in tokenizer.batch_decode(output, skip_special_tokens=True))
            LOG.info("Translated %s/%s unique sources", len(predictions), len(items))

    if len(predictions) != len(items):
        raise RuntimeError("Number of predictions does not match the unique test sources")
    return predictions


def compute_metrics(predictions: list[str], reference_lists: list[list[str]]) -> dict[str, Any]:
    """SacreBLEU metrics with multiple, possibly uneven, reference counts."""
    if not predictions or len(predictions) != len(reference_lists):
        raise ValueError("Predictions and references must be nonempty and have the same length")
    if any(not refs for refs in reference_lists):
        raise ValueError("Every test source must have at least one reference")

    # SacreBLEU uses *reference-major* input: N reference streams x M samples.
    # Missing references are represented with None; never duplicate a reference
    # just to make the list lengths equal.
    max_refs = max(len(refs) for refs in reference_lists)
    reference_streams = [
        [refs[k] if k < len(refs) else None for refs in reference_lists]
        for k in range(max_refs)
    ]
    metrics = {
        "sacrebleu": BLEU(),
        "chrf": CHRF(word_order=0),
        "chrf_plus_plus": CHRF(word_order=2),
        "ter": TER(),
    }
    scores = {}
    signatures = {}
    for name, metric in metrics.items():
        scores[name] = float(metric.corpus_score(predictions, reference_streams).score)
        signatures[name] = str(metric.get_signature())
    return {"scores": scores, "metric_signatures": signatures}


def log_evaluation(
    run,
    config: dict,
    items: list[dict],
    predictions: list[str],
    report: dict[str, Any],
) -> None:
    """Log comparison-friendly scores, a preview Table, and full audit artifact."""
    run.log({f"test/{key}": value for key, value in report["scores"].items()})
    for key in (
        "original_test_rows", "invalid_test_rows_excluded",
        "unique_sources_evaluated", "multi_reference_sources",
        "model_revision", "dataset_revision", "model_id", "dataset_id", "direction",
    ):
        run.summary[key] = report[key]
    run.summary["metric_signatures"] = report["metric_signatures"]

    n_rows = min(config["tracking"].get("preview_rows", 100), len(items))
    if n_rows:
        preview = wandb.Table(columns=[
            "source", "prediction", "references", "reference_count", "dataset_row_indices"
        ])
        for item, pred in zip(items[:n_rows], predictions[:n_rows]):
            preview.add_data(
                item["source"], pred, " | ".join(item["references"]),
                len(item["references"]), ", ".join(map(str, item["row_indices"])),
            )
        run.log({"test/predictions_preview": preview})

    if config["tracking"].get("log_predictions_artifact", True):
        # Files are temporary *output artifacts*, not local model/run inputs.
        with tempfile.TemporaryDirectory(prefix="translation_eval_") as directory:
            output_dir = Path(directory)
            metrics_file = output_dir / "metrics.json"
            metrics_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            config_file = output_dir / "evaluation.yaml"
            config_file.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            predictions_file = output_dir / "predictions.jsonl"
            with predictions_file.open("w", encoding="utf-8") as stream:
                for item, prediction in zip(items, predictions):
                    stream.write(json.dumps(
                        {"source": item["source"], "prediction": prediction,
                         "references": item["references"], "row_indices": item["row_indices"]},
                        ensure_ascii=False,
                    ) + "\n")

            artifact = wandb.Artifact(
                name=f"test-predictions-{run.id}",
                type="evaluation",
                metadata={
                    "model_id": report["model_id"],
                    "model_revision": report["model_revision"],
                    "dataset_id": report["dataset_id"],
                    "dataset_revision": report["dataset_revision"],
                    "direction": report["direction"],
                    "split": "test",
                },
            )
            artifact.add_file(str(metrics_file))
            artifact.add_file(str(config_file))
            artifact.add_file(str(predictions_file))
            # Wait for upload to finish before the temporary directory vanishes.
            run.log_artifact(artifact).wait()


def evaluate_from_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_evaluation_configuration(config_path)
    hf_token = os.getenv("HF_TOKEN")  # optional for public repos, needed for private repos
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])

    tracking = config["tracking"]
    model_id = config["model"]["model_id"]
    src_lang, tgt_lang = config["model"]["src_lang"], config["model"]["tgt_lang"]
    run_name = tracking.get("run_name") or f"test_{model_id.split('/')[-1]}_{src_lang}_to_{tgt_lang}"
    run = wandb.init(
        project=tracking["project"],
        entity=tracking.get("entity"),
        group=tracking.get("group"),
        name=run_name,
        job_type="evaluation",
        tags=["evaluation", "test", f"{src_lang}->{tgt_lang}"],
        config=config,
    )
    if run is None:
        raise RuntimeError("W&B did not initialize a run; check WANDB_MODE and credentials")

    exit_code = 1
    try:
        api = HfApi(token=hf_token)
        model_sha = api.model_info(model_id, revision=config["model"].get("revision")).sha
        dataset_id = config["data"]["dataset_id"]
        dataset_sha = api.dataset_info(dataset_id, revision=config["data"].get("revision")).sha
        LOG.info("Model: %s @ %s", model_id, model_sha)
        LOG.info("Test dataset: %s @ %s", dataset_id, dataset_sha)

        test = load_dataset(
            dataset_id, split=config["data"]["split"],
            revision=dataset_sha, token=hf_token,
        )
        src_col = config["data"]["source_column"]
        tgt_col = config["data"]["target_column"]
        for column in (src_col, tgt_col):
            if column not in test.column_names:
                raise ValueError(f"Column {column!r} not found in dataset: {test.column_names}")

        items, invalid = group_test_references(test, src_col, tgt_col)
        if not items:
            raise ValueError("The test split contains no valid source/reference pairs")
        LOG.info("Test: %d raw rows; %d unique sources; %d invalid pairs",
                 len(test), len(items), invalid)

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_sha, token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=model_sha, token=hf_token)
        target_id = configure_languages(model, tokenizer, src_lang, tgt_lang)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        LOG.info("Evaluation device: %s", device)

        predictions = generate_translations(model, tokenizer, items, config["generation"], target_id)
        report = compute_metrics(predictions, [item["references"] for item in items])
        report.update({
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "model_revision": model_sha,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_sha,
            "direction": f"{src_lang}->{tgt_lang}",
            "split": "test",
            "original_test_rows": len(test),
            "invalid_test_rows_excluded": invalid,
            "unique_sources_evaluated": len(items),
            "multi_reference_sources": sum(len(item["references"]) > 1 for item in items),
            "generation": config["generation"],
            "wandb_run_url": run.url,
        })
        log_evaluation(run, config, items, predictions, report)
        run.summary["status"] = "completed"
        LOG.info("W&B evaluation: %s", run.url)
        LOG.info("Test metrics: %s", json.dumps(report["scores"], indent=2))
        exit_code = 0
        return report
    except Exception as exc:
        LOG.exception("Evaluation failed")
        run.summary["status"] = "failed"
        run.summary["error"] = str(exc)
        raise
    finally:
        wandb.finish(exit_code=exit_code)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_from_config(CONFIG_PATH)


if __name__ == "__main__":
    main()