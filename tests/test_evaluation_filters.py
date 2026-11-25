import pytest
from datasets import Dataset

from evaluate_models import apply_row_filters


def corpus():
    return Dataset.from_dict({
        "original_sentence": ["N sta bon", "Teste", "Bu sta dretu?"],
        "translation": ["Estou bem", "Test", "Estás bem?"],
        "target_iso_code": ["pt-pt", "en-us", "pt-pt"],
    })


def test_keeps_only_rows_matching_the_filter():
    kept = apply_row_filters(corpus(), {"target_iso_code": "pt-pt"})
    assert kept["original_sentence"] == ["N sta bon", "Bu sta dretu?"]


def test_unknown_filter_column_is_rejected():

    with pytest.raises(KeyError, match="translation_iso_code.*available.*target_iso_code"):
        apply_row_filters(corpus(), {"translation_iso_code": "pt-pt"})
