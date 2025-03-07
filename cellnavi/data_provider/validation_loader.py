import os
import random
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import ValidationDataset


class ValidationSampler:
    def __init__(self, end_iter):
        self.end_iter = end_iter
        pass

    def __iter__(self):
        i = 0
        while True:
            if i == self.end_iter:
                return
            else:
                yield i
                i += 1


class ValidationLoader(DataLoader):
    def __init__(self):
        self._dataset = ValidationDataset()
        self._sampler = ValidationSampler(end_iter=len(self._dataset))
        super().__init__(
            self._dataset,
            batch_size=None,
            sampler=self._sampler,
            num_workers=1,
            pin_memory=True,
        )

    def __len__(self):
        return len(self._dataset)
