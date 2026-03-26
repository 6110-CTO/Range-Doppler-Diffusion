import os
import json
from types import SimpleNamespace


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return SimpleNamespace(**config_dict)
