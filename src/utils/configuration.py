import yaml


def load_training_configuration(path: str) -> dict:
    """Load a YAML training configuration from ``path``."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {path}") from None
    if config is None:
        raise ValueError(f"Config file at {path} is empty")
    if not isinstance(config, dict):
        raise TypeError(f"Config file at {path} must contain a mapping, got {type(config).__name__}")
    return config


def validate_training_config(config: dict) -> None:
    """Raise if the training configuration is missing required settings.

    Checks only the schema shared by manual training and HPO; workflow-specific
    rules (search space, GPU capabilities) belong to their own validators.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping, got {type(config).__name__}")

    required = {
        "model": ("model_name", "src_lang", "tgt_lang"),
        "data": ("dataset", "max_length"),
        "tracking": ("output_base_dir",),
    }
    for section, keys in required.items():
        value = config.get(section)
        if not isinstance(value, dict):
            raise KeyError(f"Missing config section: {section}")
        for key in keys:
            if value.get(key) in (None, ""):
                raise KeyError(f"Missing config value: {section}.{key}")

    if not isinstance(config.get("training_arguments"), dict):
        raise KeyError("Missing config section: training_arguments")

    patience = config.get("early_stopping_patience")
    if patience is not None and not isinstance(patience, int):
        raise TypeError(f"early_stopping_patience must be an int or null, got {type(patience).__name__}")
