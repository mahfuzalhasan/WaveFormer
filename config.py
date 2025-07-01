import os
import yaml

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            setattr(self, key, value)

# Usage: from config import config
config = Config() 