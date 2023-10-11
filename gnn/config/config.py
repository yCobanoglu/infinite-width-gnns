import os
import pprint
import uuid
from copy import deepcopy
from pathlib import Path

import yaml
from yaml import Loader

from gnn.config.data_config import DataConfig
from gnn.config.model_config import ModelConfig


class Config:
    def __init__(self, config, log_path=None):
        self._id = str(id(self))
        self.run = None
        self._config = config
        self.learning_task = config["learning_task"]
        self.runs = config.get("runs", 1)
        self.parallel = config.get("parallel", 1)
        self._log_path = log_path
        self.model = ModelConfig(**config["model"], _id=self._id)
        self.data = DataConfig(**config["data"], _id=self._id)
        self.loss = config["loss"] if config.get("loss") else None
        self.accuracy = config["accuracy"] if config.get("accuracy") else None
        self.config_id = f"{self.model.name}-{self.data.name}-{self._id}"

        self.metrics = None

    def set_metrics(self, metrics):
        self.metrics = metrics

    def clone_reset_name(self):
        return Config(deepcopy(self._config), self._log_path)

    def to_log_dict(self):
        return self._config

    @property
    def log_dir(self):
        return Path(f"{self._log_path}/{self.config_id}")

    @classmethod
    def from_file(cls, path, log_path=None):
        return cls(Config._read_config(path), log_path)

    def __str__(self):
        return pprint.pformat(self.config)

    @staticmethod
    def _read_config(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=Loader)
        return config

    def save_config_to_dir(self, dir):
        if not dir.exists():
            os.makedirs(dir)
        with open(f"{dir}/config.yaml", "w") as outfile:
            yaml.dump(self._config, outfile)

    def save_config_to_log_dir(self):
        self.save_config_to_dir(self.log_dir)
