import yaml

def load_training_configuration(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        raise FileNotFoundError(f"Config file not found at {path}")

    return config

def test_configuration(config: dict):
    """
    Function that validates the needed parameters for the workflow
    :param config:
    :return:
    """
    must_be_parameters = ["model", "data", "tracking"]
    for parameter in must_be_parameters:
        if parameter not in config:
            raise KeyError(f"Parameter {parameter} not found in config file")

