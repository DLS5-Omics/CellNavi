import os
import hashlib
import functools
from pathlib import Path

import yaml
import numpy as np

from .util import ensure_dir
from .logger import Logger

__all__ = ["config"]

#save_path = f"/home/PATH_TO_SAVE"
save_path = f"/home/pany3/pany3/CellNavi/dataset_full/"
#data_path = f"/home/PATH_TO_DATA"
data_path = f"/home/pany3/pany3/CellNavi/dataset_full/"

class Config:
    def __init__(self):
        # for data
        self.global_batch_size = 128
        self.local_batch_size = 1
        self.n_cls = 2058
        self.bins = np.linspace(0, 9.3, 65)

        # for training
        self.mixed_precision = True
        self.nr_step = 3000
        self.warmup_step = 500
        try:
            import torch

            self.n_gpu = torch.distributed.get_world_size()
        except:
            self.n_gpu = 1
        self.n_accumulate = (
            self.global_batch_size // self.n_gpu // self.local_batch_size
        )
        self.lr = 0.001

        # for checkpoint
        self.chk_time_interval = 3600
        self.chk_step_interval = [100]


    def _create_logger(self, path, **kwargs):
        return Logger(path, **kwargs)

    @property
    @functools.lru_cache(maxsize=1)
    def train_logger(self):
        return self._create_logger(self.log_dir / "train_log.txt")

    @property
    @functools.lru_cache(maxsize=1)
    def dataset_dir(self):
        path = Path(data_path)
        if path.exists():
            return path
        raise Exception("DatasetNotFoundError")

    @property
    @ensure_dir
    def saved_dir(self):
        return Path(save_path)

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def log_dir(self):
        return self.saved_dir / "log"

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def model_dir(self):
        return self.saved_dir / "finetune" / "model"

    @property
    @functools.lru_cache(maxsize=1)
    def pretrain_model_dir(self):
        return self.saved_dir / "pretrain" / "model"
    
    @property
    @functools.lru_cache(maxsize=1)
    def checkpoint_dir(self):
        return self.model_dir


config = Config()
