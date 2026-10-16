import os
from dotenv import load_dotenv

import re
import unicodedata
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi


def normalize_sentence(sentence):
    if not isinstance(sentence, str):
        return None

    sentence = unicodedata.normalize("NFKC", sentence)
    sentence = sentence.casefold()
    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence or None


def get_overlap_mask(df, reference_df):
    """
    Detects KEA and PT sentence overlaps between two datasets.
    """
    reference_kea = set(reference_df["kea_norm"].dropna())
    reference_pt = set(reference_df["pt_norm"].dropna())

    kea_overlap = df["kea_norm"].isin(reference_kea)
    pt_overlap = df["pt_norm"].isin(reference_pt)

    return kea_overlap | pt_overlap


def print_overlap_statistics(df, overlap_mask, name):
    """
    Prints overlap statistics for a dataset.
    """
    print(f"\n{name}")
    print(f"Original samples: {len(df)}")
    print(f"Samples to remove: {overlap_mask.sum()}")
    print(f"Percentage: {overlap_mask.mean() * 100:.2f}%")
    print(f"Remaining: {(~overlap_mask).sum()}")


def data_cleaning(dataset):
    """
    Removes sentence overlaps between train, validation and test.

    Priority:
        1. Preserve official test
        2. Preserve validation
        3. Clean train
    """

    # 1. Prepare datasets
    dataframes = {}

    for split, ds in dataset.items():
        df = ds.to_pandas()
        df["kea_norm"] = df["original_sentence"].apply(normalize_sentence)
        df["pt_norm"] = df["translation"].apply(normalize_sentence)

        dataframes[split] = df

    train_df = dataframes["train"]
    val_df = dataframes["validation"]
    test_df = dataframes["test"]

    # 2. Detect overlaps with official test
    train_test_mask = get_overlap_mask(train_df, test_df)
    val_test_mask = get_overlap_mask(val_df, test_df)

    print_overlap_statistics(train_df, train_test_mask, "TRAIN -> TEST")
    print_overlap_statistics(val_df, val_test_mask, "VALIDATION -> TEST")

    # 3. Remove overlaps with official test
    train_clean = train_df[~train_test_mask].copy()
    val_clean = val_df[~val_test_mask].copy()

    # 4. Remove train overlaps with validation
    train_val_mask = get_overlap_mask(train_clean, val_clean)
    print_overlap_statistics(train_clean, train_val_mask,"TRAIN -> VALIDATION")
    train_final = train_clean[~train_val_mask].copy()

    # 5. Validate final datasets
    assert not get_overlap_mask(train_final, val_clean).any()
    assert not get_overlap_mask(train_final, test_df).any()
    assert not get_overlap_mask(val_clean, test_df).any()

    # 6. Convert to Hugging Face DatasetDict
    cleaned_dataset = DatasetDict({
        "train": dataset["train"].select(train_final.index.tolist()),
        "validation": dataset["validation"].select(val_clean.index.tolist()),
        "test": dataset["test"]
    })

    print("\nFinal dataset:")
    print(cleaned_dataset)

    return cleaned_dataset


def main():
    dataset_source = "Iscte-Sintra/Korpora_paralela_st"
    dataset_clean = "Iscte-Sintra/Korpora_paralela_st_clean"
    # Load environment variables
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")

    # Load original dataset
    dataset = load_dataset(dataset_source, token=hf_token)
    # Remove data leakage
    cleaned_dataset = data_cleaning(dataset)

    # Upload cleaned dataset
    cleaned_dataset.push_to_hub(
        dataset_clean,
        token=hf_token,
        private=True,
        commit_message="Cleaned dataset with leakage removal"
    )

    # Retrieve dataset revision
    api = HfApi(token=hf_token)
    revision = api.dataset_info(dataset_clean).sha

    print(f"Dataset published: {dataset_clean}")
    print(f"Revision: {revision}")


if __name__ == "__main__":
    main()