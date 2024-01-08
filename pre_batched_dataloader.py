from pathlib import Path
from typing import Callable, Any
import yaml

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import pickle

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset

# Local imports:
from data_loaders_iterable import get_saved_train_loader, get_saved_test_loader, get_saved_test_cross_sub_loader

BATCH_SIZE = 128

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")

PRE_BATCHED_TENSORS_PATH = PATH_TO_SUBSET1.joinpath("pre-batched-tensors", f"bs{BATCH_SIZE}")


def create_pre_batched_tensors(batch_size=BATCH_SIZE):
    train_loader = get_saved_train_loader(batch_size)
    test_loader = get_saved_test_loader(batch_size)
    cross_test_loader = get_saved_test_cross_sub_loader(batch_size)

    train_tensors_path = PRE_BATCHED_TENSORS_PATH.joinpath("train")
    test_tensors_path = PRE_BATCHED_TENSORS_PATH.joinpath("test")
    cross_test_tensors_path = PRE_BATCHED_TENSORS_PATH.joinpath("cross-test")
    train_tensors_path.mkdir(parents=True, exist_ok=True)
    test_tensors_path.mkdir(parents=True, exist_ok=True)
    cross_test_tensors_path.mkdir(parents=True, exist_ok=True)

    print("Saving batches from train loader:")
    batches = len(train_loader)
    for (i, item) in tqdm(enumerate(iter(train_loader)), total=batches):
        batch_path = train_tensors_path.joinpath(f"batch-{i}")
        batch_path.mkdir(exist_ok=True)
        X_path = batch_path.joinpath("X.pt")
        y_path = batch_path.joinpath("y.pt")
        X, y = item
        torch.save(X, X_path)
        torch.save(y, y_path)

    batches = len(test_loader)
    for (i, item) in tqdm(enumerate(iter(test_loader)), total=batches):
        batch_path = test_tensors_path.joinpath(f"batch-{i}")
        batch_path.mkdir(exist_ok=True)
        X_path = batch_path.joinpath("X.pt")
        y_path = batch_path.joinpath("y.pt")
        X, y = item
        torch.save(X, X_path)
        torch.save(y, y_path)

    batches = len(cross_test_loader)
    for (i, item) in tqdm(enumerate(iter(cross_test_loader)), total=batches):
        batch_path = cross_test_tensors_path.joinpath(f"batch-{i}")
        batch_path.mkdir(exist_ok=True)
        X_path = batch_path.joinpath("X.pt")
        y_path = batch_path.joinpath("y.pt")
        X, y = item
        torch.save(X, X_path)
        torch.save(y, y_path)


if __name__ == "__main__":
    create_pre_batched_tensors(batch_size=BATCH_SIZE)
