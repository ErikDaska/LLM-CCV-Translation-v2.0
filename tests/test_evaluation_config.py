import pytest
import yaml

from evaluate_models import load_evaluation_configuration


def write_config(tmp_path, filters):
    config = {
        "model": {"model_id": "Iscte-Sintra/nllb600M-kea-pt-bidirectional",
                  "src_lang": "kea_st_XX", "tgt_lang": "por_Latn"},
        "data": {"dataset_id": "Iscte-Sintra/Korpora_paralela_st_clean", "split": "test",
                 "source_column": "original_sentence", "target_column": "translation"},
        "generation": {"batch_size": 16, "max_input_length": 128,
                       "max_new_tokens": 128, "num_beams": 4},
        "tracking": {"project": "Translation Models"},
    }
    if filters is not None:
        config["data"]["filters"] = filters
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_filters_mapping_is_accepted(tmp_path):
    config = load_evaluation_configuration(write_config(tmp_path, {"target_iso_code": "pt-pt"}))
    assert config["data"]["filters"] == {"target_iso_code": "pt-pt"}


@pytest.mark.parametrize("filters", [["target_iso_code", "pt-pt"], {"target_iso_code": 1}])
def test_malformed_filters_are_rejected(tmp_path, filters):
    with pytest.raises(ValueError, match="data.filters"):
        load_evaluation_configuration(write_config(tmp_path, filters))
