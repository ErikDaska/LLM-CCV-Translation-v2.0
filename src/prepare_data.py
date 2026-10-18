"""Clean a KEA-PT parallel corpus while preserving N:N translation relations.

Rules:
- Within a split: remove only repeated normalized (KEA, PT) translation pairs.
- Across splits: remove a lower-priority row when normalized KEA OR PT is equal.
- Across splits: additionally remove a lower-priority row when normalized KEA OR PT
  meets the configurable Levenshtein similarity threshold.

Priority: test > validation > train. All detected duplicate groups and all removed
rows are recorded in JSON files, with separate reports for each cleaning stage.
"""

import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sympy import true

PAIR_COLUMNS = ["kea_norm", "pt_norm"]
LANGUAGES = {
    "kea": ("kea_norm", "original_sentence"),
    "pt": ("pt_norm", "translation"),
}

# Change this value to control automatic near-duplicate removal.
# Example: 0.95 = remove when KEA OR PT similarity is >= 95%.
SIMILARITY_THRESHOLD = 0.90

# Remove exact repeated translation pairs from test as well. Set False if the
# official test split must remain untouched for benchmark reproducibility.
CLEAN_TEST_DUPLICATES = True

# Keep local results; publish only when explicitly enabled.
SAVE_CLEANED_DATASET_LOCALLY = True
PUSH_DATASET_TO_HUB = True
UPLOAD_REPORTS_TO_HUB = False

DATASET_SOURCE = "Iscte-Sintra/Korpora_paralela_st"
DATASET_DESTINATION = "Iscte-Sintra/Korpora_paralela_st_clean"


def normalize_sentence(sentence):
    """Normalize Unicode, case and whitespace without deleting punctuation."""
    if not isinstance(sentence, str):
        return None
    sentence = unicodedata.normalize("NFKC", sentence).casefold()
    return re.sub(r"\s+", " ", sentence).strip() or None


def normalize_splits(dataset):
    frames = {}
    for split, ds in dataset.items():
        frame = ds.to_pandas()
        frame["kea_norm"] = frame["original_sentence"].apply(normalize_sentence)
        frame["pt_norm"] = frame["translation"].apply(normalize_sentence)
        frames[split] = frame
    return frames


def save_audit(df, path):
    """Save every selected original row and the reason/metadata as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = df.copy()
    audit.insert(0, "source_row_index", audit.index.to_numpy())
    audit.to_json(path, orient="records", force_ascii=False, indent=2, date_format="iso")


def clean_internal_duplicates(df, split, report_dir, remove=True):
    """Only identical COMPLETE (KEA, PT) pairs are internal duplicates."""
    complete = df[PAIR_COLUMNS].notna().all(axis=1)
    duplicate_extra = pd.Series(False, index=df.index)
    duplicate_group = pd.Series(False, index=df.index)

    duplicate_extra.loc[complete] = df.loc[complete].duplicated(subset=PAIR_COLUMNS, keep="first")
    duplicate_group.loc[complete] = df.loc[complete].duplicated( subset=PAIR_COLUMNS, keep=False)

    audit = df.loc[duplicate_group].copy()
    audit["reason"] = "internal_duplicate_pair"
    audit["source_split"] = split
    audit["status"] = [
        "removed" if remove and bool(duplicate_extra.loc[idx]) else "retained"
        for idx in audit.index
    ]

    filename = f"internal_duplicates_{split}.json"
    save_audit(audit, report_dir / filename)

    cleaned = df.loc[~duplicate_extra].copy() if remove else df.copy()
    stats = {
        "original": int(len(df)),
        "repeated_unique_pairs": int( df.loc[duplicate_group, PAIR_COLUMNS].drop_duplicates().shape[0]),
        "repeated_rows_including_first": int(duplicate_group.sum()),
        "extra_occurrences": int(duplicate_extra.sum()),
        "removed": int(duplicate_extra.sum()) if remove else 0,
        "rows_with_missing_sentence": int((~complete).sum()),
        "remaining": int(len(cleaned)),
        "report": filename,
    }
    print( f"INTERNAL {split.upper()}: {stats['extra_occurrences']} repeated occurrences, {stats['removed']} removed; {stats['remaining']} remaining")
    return cleaned, stats


def overlap_flags(df, reference_df):
    """Identify exact cross-split matches in KEA OR PT."""
    reference_kea = set(reference_df["kea_norm"].dropna())
    reference_pt = set(reference_df["pt_norm"].dropna())
    return (
        df["kea_norm"].isin(reference_kea),
        df["pt_norm"].isin(reference_pt),
    )


def get_overlap_mask(df, reference_df):
    kea_match, pt_match = overlap_flags(df, reference_df)
    return kea_match | pt_match


def remove_cross_split_overlap(df, reference_df, source, reference, report_dir):
    """Remove exact cross-split leakage, with a JSON record for each removal."""
    kea_match, pt_match = overlap_flags(df, reference_df)
    remove_mask = kea_match | pt_match

    audit = df.loc[remove_mask].copy()
    audit["reason"] = "exact_cross_split_leakage"
    audit["source_split"] = source
    audit["reference_split"] = reference
    audit["matches_kea"] = kea_match.loc[remove_mask]
    audit["matches_pt"] = pt_match.loc[remove_mask]
    audit["status"] = "removed"

    filename = f"overlap_{source}_vs_{reference}.json"
    save_audit(audit, report_dir / filename)

    cleaned = df.loc[~remove_mask].copy()
    stats = {
        "original": int(len(df)),
        "overlap_kea": int(kea_match.sum()),
        "overlap_pt": int(pt_match.sum()),
        "overlap_both": int((kea_match & pt_match).sum()),
        "removed": int(remove_mask.sum()),
        "remaining": int(len(cleaned)),
        "report": filename,
    }
    print(f"EXACT {source.upper()} -> {reference.upper()}: {stats['removed']} removed; {stats['remaining']} remaining")
    return cleaned, stats


def _best_matches_by_sentence(source_sentences, reference_df, norm_col, threshold):
    """Find the strongest qualifying reference for every UNIQUE source sentence.

    Reference sentences are indexed by length. Length-based filtering is safe
    for normalized Levenshtein similarity: a sentence with length n can reach
    threshold t only with reference length in [ceil(t*n), floor(n/t)].
    This avoids comparing against impossible-length candidates; it is exact
    (no approximate-nearest-neighbor false negatives), but can still be costly
    for very large datasets or low similarity thresholds.
    """
    reference = reference_df.dropna(subset=[norm_col]).drop_duplicates( subset=[norm_col], keep="first")
    reference_lookup = dict(zip(reference[norm_col], reference.index))
    by_length = defaultdict(list)
    for sentence in reference_lookup:
        by_length[len(sentence)].append(sentence)

    result = {}
    for sentence in source_sentences:
        if not isinstance(sentence, str) or not sentence:
            continue

        # All exact overlaps should have been handled by the previous stage.
        # We still allow equality here if the function is used independently.
        if sentence in reference_lookup:
            result[sentence] = (sentence, 1.0, reference_lookup[sentence])
            continue

        source_length = len(sentence)
        minimum = math.ceil(threshold * source_length - 1e-10)
        maximum = math.floor(source_length / threshold + 1e-10)
        choices = []
        for length in range(minimum, maximum + 1):
            choices.extend(by_length.get(length, ()))
        if not choices:
            continue

        match = process.extractOne( sentence, choices, scorer=Levenshtein.normalized_similarity, score_cutoff=threshold)
        if match is None:
            continue

        matched_sentence, score, _ = match
        result[sentence] = ( matched_sentence, float(score), int(reference_lookup[matched_sentence]))

    return result


def remove_similar_cross_split(
    df,
    reference_df,
    source,
    reference,
    report_dir,
    threshold=SIMILARITY_THRESHOLD,
):
    """Remove rows where KEA OR PT has Levenshtein similarity >= threshold.

    If both languages match, the source row is removed just ONCE and both
    matches are recorded. The reference split is NEVER modified.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1].")

    # source index -> {"kea": {...}, "pt": {...}}
    matches_by_row = {}
    count_by_language = {}

    for language, (norm_col, original_col) in LANGUAGES.items():
        unique_sources = df[norm_col].dropna().unique()
        best_matches = _best_matches_by_sentence(unique_sources, reference_df, norm_col, threshold)
        language_rows = 0

        for idx, sentence in df[norm_col].items():
            match = best_matches.get(sentence)
            if match is None:
                continue

            reference_sentence, score, reference_index = match
            reference_row = reference_df.loc[reference_index]
            matches_by_row.setdefault(idx, {})[language] = {
                "similarity": round(score, 6),
                "source_sentence": df.at[idx, original_col],
                "reference_sentence": reference_row[original_col],
                "normalized_source": sentence,
                "normalized_reference": reference_sentence,
                "reference_row_index": int(reference_index),
                # Entire reference row for fully reproducible audits.
                "reference_row": reference_row.to_dict(),
            }
            language_rows += 1

        count_by_language[language] = language_rows

    removed_indices = list(matches_by_row)
    audit = df.loc[removed_indices].copy()
    audit["reason"] = "similar_cross_split_leakage"
    audit["source_split"] = source
    audit["reference_split"] = reference
    audit["similarity_threshold"] = threshold
    audit["matched_languages"] = [list(matches_by_row[idx]) for idx in removed_indices]
    audit["matches"] = [matches_by_row[idx] for idx in removed_indices]
    audit["status"] = "removed"

    filename = f"similarity_{source}_vs_{reference}.json"
    save_audit(audit, report_dir / filename)
    cleaned = df.drop(index=removed_indices).copy()

    stats = {
        "original": int(len(df)),
        "threshold": threshold,
        "metric": "Levenshtein.normalized_similarity",
        "matched_kea": count_by_language["kea"],
        "matched_pt": count_by_language["pt"],
        "matched_both": sum(len(matches) == 2 for matches in matches_by_row.values()),
        "removed": len(removed_indices),
        "remaining": len(cleaned),
        "report": filename,
    }
    print( f"SIMILAR {source.upper()} -> {reference.upper()} (threshold={threshold:.0%}): {stats['removed']} removed; {stats['remaining']} remaining")
    return cleaned, stats


def data_cleaning(
    dataset,
    report_dir,
    similarity_threshold=SIMILARITY_THRESHOLD,
    clean_test_duplicates=CLEAN_TEST_DUPLICATES,
):
    """Run cleaning in priority order: test > validation > train."""
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1].")

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frames = normalize_splits(dataset)
    summary = {
        "config": {
            "similarity_threshold": similarity_threshold,
            "similarity_metric": "Levenshtein.normalized_similarity",
            "clean_test_duplicates": clean_test_duplicates,
            "priority": ["test", "validation", "train"],
        },
        "original_sizes": {split: len(ds) for split, ds in dataset.items()},
        "internal": {},
        "cross_split": {},
        "similarity": {},
    }

    # 1. Deduplicate exact internal (KEA, PT) pairs, without losing N:N pairs.
    for split in ("train", "validation", "test"):
        frames[split], stats = clean_internal_duplicates( frames[split], split, report_dir, remove=(split != "test" or clean_test_duplicates))
        summary["internal"][split] = stats

    train_df = frames["train"]
    val_df = frames["validation"]
    test_df = frames["test"]

    # 2. Protect test against EXACT overlaps in train and validation.
    train_df, stats = remove_cross_split_overlap(train_df, test_df, "train", "test", report_dir)
    summary["cross_split"]["train_vs_test"] = stats

    val_df, stats = remove_cross_split_overlap(val_df, test_df, "validation", "test", report_dir)
    summary["cross_split"]["validation_vs_test"] = stats

    # 3. Protect test against NEAR duplicates. Clean validation first so that
    # train is later compared only against validation rows that survive.
    val_df, stats = remove_similar_cross_split(val_df, test_df, "validation", "test", report_dir, similarity_threshold)
    summary["similarity"]["validation_vs_test"] = stats

    train_df, stats = remove_similar_cross_split(train_df, test_df, "train", "test", report_dir, similarity_threshold)
    summary["similarity"]["train_vs_test"] = stats

    # 4. Protect the FINAL validation split against exact / similar train rows.
    train_df, stats = remove_cross_split_overlap(train_df, val_df, "train", "validation", report_dir)
    summary["cross_split"]["train_vs_validation"] = stats

    train_df, stats = remove_similar_cross_split(train_df, val_df, "train", "validation", report_dir, similarity_threshold)
    summary["similarity"]["train_vs_validation"] = stats

    # 5. Invariants: no remaining EXACT overlaps; no extra internal full pairs.
    assert not get_overlap_mask(train_df, val_df).any()
    assert not get_overlap_mask(train_df, test_df).any()
    assert not get_overlap_mask(val_df, test_df).any()
    for split, frame in (("train", train_df), ("validation", val_df)):
        complete = frame[PAIR_COLUMNS].notna().all(axis=1)
        assert not frame.loc[complete].duplicated(subset=PAIR_COLUMNS).any(), split
    if clean_test_duplicates:
        complete = test_df[PAIR_COLUMNS].notna().all(axis=1)
        assert not test_df.loc[complete].duplicated(subset=PAIR_COLUMNS).any(), "test"

    # 6. Preserve original positional indices from Dataset.to_pandas(),
    # allowing Dataset.select() to recover the complete original columns.
    cleaned_dataset = DatasetDict({
        "train": dataset["train"].select(train_df.index.tolist()),
        "validation": dataset["validation"].select(val_df.index.tolist()),
        "test": dataset["test"].select(test_df.index.tolist()),
    })
    summary["final_sizes"] = {
        split: int(len(cleaned_dataset[split])) for split in cleaned_dataset
    }
    summary["removed_total"] = sum(
        stats["removed"]
        for key in ("internal", "cross_split", "similarity")
        for stats in summary[key].values()
    )
    with (report_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\nFinal dataset:")
    print(cleaned_dataset)
    print(f"Total rows removed: {summary['removed_total']}")
    print(f"JSON audit reports: {report_dir.resolve()}")
    return cleaned_dataset


def main():
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path("cleaning_reports") / run_timestamp

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")

    dataset = load_dataset(DATASET_SOURCE, token=token)
    cleaned_dataset = data_cleaning(
        dataset,
        report_dir=report_dir,
        similarity_threshold=SIMILARITY_THRESHOLD,
        clean_test_duplicates=CLEAN_TEST_DUPLICATES,
    )

    if SAVE_CLEANED_DATASET_LOCALLY:
        output = report_dir / "cleaned_dataset"
        cleaned_dataset.save_to_disk(str(output))
        print(f"Cleaned dataset saved locally: {output.resolve()}")

    if PUSH_DATASET_TO_HUB:
        cleaned_dataset.push_to_hub(
            DATASET_DESTINATION,
            token=token,
            private=True,
            commit_message=("Remove exact and near-duplicate leakage across KEA-PT splits"),
        )
        api = HfApi(token=token)
        revision = api.dataset_info(DATASET_DESTINATION).sha
        print(f"Dataset published: {DATASET_DESTINATION}\nRevision: {revision}")

    if UPLOAD_REPORTS_TO_HUB:
        # Only upload JSON files, not the locally saved cleaned_dataset directory.
        api = HfApi(token=token)
        json_paths = sorted(report_dir.glob("*.json"))
        for json_path in json_paths:
            api.upload_file(
                path_or_fileobj=str(json_path),
                path_in_repo=f"cleaning_reports/{run_timestamp}/{json_path.name}",
                repo_id=DATASET_DESTINATION,
                repo_type="dataset",
                commit_message=f"Audit log: {json_path.name}",
            )
        print(f"Uploaded {len(json_paths)} JSON audit reports to Hugging Face")


if __name__ == "__main__":
    main()