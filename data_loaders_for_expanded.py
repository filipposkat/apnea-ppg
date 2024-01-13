from itertools import cycle
from pathlib import Path
from typing import Callable, Any
import yaml

import numpy as np
import pandas as pd
import random
import pickle
from sortedcontainers import SortedList

import seaborn as sn
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset

WINDOW_SAMPLES_SIZE = 512
N_SIGNALS = 2
CROSS_SUBJECT_TEST_SIZE = 100
BATCH_WINDOW_SAMPLING_RATIO = 0.1
SEED = 33
BATCH_SIZE = 256
NUM_WORKERS = 2
PREFETCH_FACTOR = 2

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1

ARRAYS_DIR = PATH_TO_SUBSET1.joinpath("arrays")
EXPANDED_ARRAYS_DIR = PATH_TO_SUBSET1.joinpath("arrays-expanded")

# Paths for saving dataloaders:
dataloaders_path = PATH_TO_SUBSET1_TRAINING.joinpath("dataloaders-for-expanded")
dataloaders_path.mkdir(parents=True, exist_ok=True)

subset_ids = [int(f.name) for f in EXPANDED_ARRAYS_DIR.iterdir() if f.is_dir()]
random.seed(SEED)
cross_test_ids = random.sample(subset_ids, CROSS_SUBJECT_TEST_SIZE)
train_ids = [id for id in subset_ids if id not in cross_test_ids]


class ExpandedDataset(Dataset):

    def __init__(self, subject_ids: list[int], dataset_split: str = "train", desired_target: str = "pleth",
                 arrays_dir: Path = EXPANDED_ARRAYS_DIR, transform=None, target_transform=None):
        self.subject_ids = subject_ids
        self.split = dataset_split
        self.arrays_dir = arrays_dir
        self.target = desired_target
        self.transform = transform
        self.target_transform = target_transform

        self.id_size_dict = {}
        self.total_windows = 0
        for id in self.subject_ids:
            n_windows = self.get_n_windows_by_id(id)
            self.id_size_dict[id] = n_windows
            self.total_windows += n_windows

    def __len__(self):
        # Find out the total windows:
        self.total_windows = 0
        for id in self.subject_ids:
            self.total_windows += self.id_size_dict[id]

        return self.total_windows

    def get_n_windows_by_id(self, subject_id: int) -> int:
        if self.split == "train":
            sub_dir = EXPANDED_ARRAYS_DIR.joinpath(str(subject_id).zfill(4), "train")
        else:
            sub_dir = EXPANDED_ARRAYS_DIR.joinpath(str(subject_id).zfill(4), "test")

        dir_iter = sub_dir.iterdir()
        n_windows = 0
        for fpath in dir_iter:
            if fpath.is_file():
                n_windows += 1
        return n_windows

    def window_loader(self, sub_id: int, window_id: int) -> tuple[np.array, np.array]:
        if self.split == "train":
            X_path = self.arrays_dir.joinpath(str(sub_id).zfill(4), "train", f"X_{window_id}.npy")
            y_path = self.arrays_dir.joinpath(str(sub_id).zfill(4), "train", f"y_{window_id}.npy")
        else:
            X_path = self.arrays_dir.joinpath(str(sub_id).zfill(4), "test", f"X_{window_id}.npy")
            y_path = self.arrays_dir.joinpath(str(sub_id).zfill(4), "test", f"y_{window_id}.npy")

        X = np.load(str(X_path)).astype("float32")  # shape: (window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=0, axis2=1)  # shape after swap: (n_signals, window_size), Flow comes first

        if "flow" == self.target:
            y = X[0, :]
        else:
            y = np.load(str(y_path)).ravel().astype("uint8")

        # Drop the flow signal since only Pleth will be used as input:
        X = np.delete(X, 0, axis=0)
        return X, y

    def __getitem__(self, subject_window_index: tuple[int, int]) -> tuple[Any, Any]:
        """
        :param idx: Index that will be mapped to (subject_id, window_id) based on index_map
        :return: (signal, labels) where signal = Pleth signal of shape (
        """

        subject_id = subject_window_index[0]
        window_id = subject_window_index[1]
        X, y = self.window_loader(subject_id, window_id)

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y


class BatchSampler(Sampler[list[int]]):
    def __init__(self, subject_ids: list[int], batch_size: int, id_size_dict: dict[int: int],
                 shuffle=False, seed=None) -> None:
        """
        :param subject_ids:
        :param batch_size:
        :param id_size_dict:
        :param shuffle: Whether to shuffle the subjects between epochs
        :param seed: The seed to use for shuffling and sampling
        """

        super().__init__()
        self.subject_ids = subject_ids
        self.batch_size = batch_size

        self.shuffle = shuffle
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        self.id_size_dict = id_size_dict

    def __len__(self) -> int:
        # Find out the total windows:
        self.total_windows = 0
        for id in self.subject_ids:
            self.total_windows += self.id_size_dict[id]

        # Number of batches:
        return (self.total_windows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = 0
        # by default tuples are sorted with preference given to the first elements (subject in this case):
        used_2d_indices = SortedList()

        # Shuffle ids in-place:
        if self.shuffle:
            self.rng.shuffle(self.subject_ids)

        # Cyclic iterator of our ids:
        pool = cycle(self.subject_ids)
        while batches < len(self) - 1:
            sub_id1 = next(pool)
            sub_id2 = next(pool)
            sub_id3 = next(pool)

            ids_tmp = [sub_id1, sub_id2, sub_id3]

            # Get the number of windows in the third subject, this will be needed to calculate the last index
            n_windows1 = self.id_size_dict[sub_id1]
            n_windows2 = self.id_size_dict[sub_id2]
            n_windows3 = self.id_size_dict[sub_id3]

            indices1 = [(sub_id1, i) for i in range(n_windows1) if (sub_id1, i) not in used_2d_indices]
            indices2 = [(sub_id2, i) for i in range(n_windows2) if (sub_id2, i) not in used_2d_indices]
            indices3 = [(sub_id3, i) for i in range(n_windows3) if (sub_id3, i) not in used_2d_indices]

            combined_indices = [*indices1, *indices2, *indices3]
            n_combined_indices = len(combined_indices)

            # Check if the windows available for sampling are enough for a batch:
            while n_combined_indices < self.batch_size:
                # If three subjects do not have enough windows then use more:
                next_sub_id = next(pool)

                # Get the number of windows in the third subject, this will be needed to calculate the last index
                n_windows = self.id_size_dict[next_sub_id]

                indices_to_add = [(next_sub_id, i) for i in range(n_windows) if (next_sub_id, i) not in used_2d_indices]
                combined_indices.extend(indices_to_add)
                ids_tmp.append(next_sub_id)

            # Sample windows:
            batch_2d_indices = self.rng.sample(combined_indices, self.batch_size)

            if len(batch_2d_indices) != self.batch_size:
                print("Indices less than batch size")

            yield batch_2d_indices

            # batch_1d_indices = [self.index_map_inverse[two_dim_index] for two_dim_index in batch_2d_indices]
            used_2d_indices.update(batch_2d_indices)
            batches += 1

        # The last batch may contain less than the batch_size:
        unused_2d_indices = []
        for sub_id in self.subject_ids:
            n_windows = self.id_size_dict[sub_id]
            for window_index in range(n_windows):
                if (sub_id, window_index) not in used_2d_indices:
                    unused_2d_indices.append((sub_id, window_index))

        assert len(unused_2d_indices) <= self.batch_size
        yield unused_2d_indices


def get_new_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR) -> DataLoader:
    dataset = ExpandedDataset(subject_ids=train_ids,
                              dataset_split="train",
                              transform=torch.from_numpy,
                              target_transform=torch.from_numpy)

    sampler = BatchSampler(subject_ids=train_ids, batch_size=batch_size, id_size_dict=dataset.id_size_dict,
                           shuffle=True, seed=SEED)
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch)


def get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR,
                           arrays_dir: Path = None) \
        -> DataLoader:
    if NUM_WORKERS == 0:
        pre_fetch = None

    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Iterable_Train_Loader.pickle")
    if not object_file.exists():
        loader = get_new_train_loader(batch_size=batch_size, num_workers=num_workers, pre_fetch=pre_fetch)

        # Save train loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

    if object_file.is_file():
        with open(object_file, "rb") as f:
            loader = pickle.load(f)

        loader.num_workers = num_workers
        loader.prefetch_factor = pre_fetch
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

    return loader


def get_new_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR) -> DataLoader:
    dataset = ExpandedDataset(subject_ids=train_ids,
                              dataset_split="test",
                              transform=torch.from_numpy,
                              target_transform=torch.from_numpy)

    sampler = BatchSampler(subject_ids=train_ids, batch_size=batch_size, id_size_dict=dataset.id_size_dict,
                           shuffle=True, seed=SEED)
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch)


def get_saved_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR,
                          arrays_dir: Path = None) \
        -> DataLoader:
    if NUM_WORKERS == 0:
        pre_fetch = None

    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Iterable_Test_Loader.pickle")
    if not object_file.exists():
        loader = get_new_test_loader(batch_size=batch_size, num_workers=num_workers, pre_fetch=pre_fetch)

        # Save test loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

    if object_file.is_file():
        with open(object_file, "rb") as f:
            loader = pickle.load(f)

        loader.num_workers = num_workers
        loader.prefetch_factor = pre_fetch
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

    return loader


def get_new_test_cross_sub_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                  pre_fetch=PREFETCH_FACTOR) -> DataLoader:
    dataset = ExpandedDataset(subject_ids=cross_test_ids,
                              dataset_split="cross_test",
                              transform=torch.from_numpy,
                              target_transform=torch.from_numpy)

    sampler = BatchSampler(subject_ids=train_ids, batch_size=batch_size, id_size_dict=dataset.id_size_dict,
                           shuffle=True, seed=SEED)
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch)


def get_saved_test_cross_sub_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR,
                                    arrays_dir: Path = None) -> DataLoader:
    if NUM_WORKERS == 0:
        pre_fetch = None

    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Iterable_TestCrossSub_Loader.pickle")
    if not object_file.exists():
        loader = get_new_test_cross_sub_loader(batch_size=batch_size, num_workers=num_workers, pre_fetch=pre_fetch)

        # Save cross test loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

    if object_file.is_file():
        with open(object_file, "rb") as f:
            loader = pickle.load(f)

        loader.num_workers = num_workers
        loader.prefetch_factor = pre_fetch
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

    return loader


if __name__ == "__main__":
    print(train_ids)
    print(cross_test_ids)

    pleth_train_loader = get_saved_train_loader()
    pleth_test_loader = get_saved_test_loader()
    pleth_test_cross_sub_loader = get_saved_test_cross_sub_loader()

    print(f"Train n batches: {len(pleth_train_loader)}")
    print(f"Test n batches: {len(pleth_test_loader)}")
    print(f"Test_cross n batches: {len(pleth_test_cross_sub_loader)}")

    for (i, item) in enumerate(pleth_train_loader):
        X, y = item
        print(f"batch: {i},  X shape: {X.shape},  y shape: {y.shape}")
        # print(X.dtype)
        # print(y)
        # memory_usage_in_bytes = X.element_size() * X.nelement() + y.element_size() * y.nelement()
        # print(f"Memory Usage: {memory_usage_in_bytes} bytes")
