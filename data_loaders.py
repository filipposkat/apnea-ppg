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

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")

EXPANDED_ARRAYS_DIR = PATH_TO_SUBSET1.joinpath("arrays-expanded")


class Dataset(Dataset):

    def __init__(self, subject_ids: list[int], window_loader: Callable[[int, int], tuple[np.array, np.array]],
                 get_n_windows_by_id: Callable[[int], int],
                 transform=None, target_transform=None):
        self.subject_ids = subject_ids
        self.load_window = window_loader
        self.transform = transform
        self.target_transform = target_transform

        # Since the inputs are indexed with two numbers (id, window_index),
        # we need a mapping from one dimensional index to two-dimensional index:
        self.index_map = {}
        self.id_size_dict = {}
        index = 0
        for id in self.subject_ids:
            n_windows = get_n_windows_by_id(id)
            self.id_size_dict[id] = n_windows
            for window_index in range(n_windows):
                self.index_map[index] = (id, window_index)
                index += 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        :param idx: Index that will be mapped to (subject_id, window_id) based on index_map
        :return: (signal, labels) where signal = Pleth signal of shape (
        """

        subject_id = self.index_map[idx][0]
        window_id = self.index_map[idx][1]
        X, y = self.load_window(subject_id, window_id)

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y


class BatchSampler(Sampler[list[int]]):
    def __init__(self, subject_ids: list[int], batch_size: int,
                 index_map: dict[int: tuple[int, int]],
                 id_size_dict: dict[int: int],
                 shuffle=False,
                 seed=None) -> None:
        """
        :param subject_ids:
        :param batch_size:
        :param index_map: Index map provided by the Dataset
        :param id_size_dict:
        :param shuffle: Whether to shuffle the subjects between epochs
        :param seed: The seed to use for shuffling and sampling
        """

        self.subject_ids = subject_ids
        # Calculate inverse index_map, which maps (subject_id, window_id) to id
        self.index_map_inverse = {v: k for k, v in index_map.items()}
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.rng = random.Random()
        if seed is not None:
            self.rng = random.Random(seed)

        self.id_size_dict = id_size_dict

    def __len__(self) -> int:
        # Number of batches:
        return (len(self.index_map_inverse) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # for batch in torch.chunk(torch.argsort(sizes), len(self)):
        #     yield batch.tolist()

        batches = 0
        used_1d_indices = SortedList()

        # Shuffle ids in-place:
        if self.shuffle:
            self.rng.shuffle(self.subject_ids)

        # Cyclic iterator of our ids:
        pool = cycle(self.subject_ids)

        while batches < len(self) - 1:
            sub_id1 = next(pool)
            sub_id2 = next(pool)
            sub_id3 = next(pool)

            # Get the number of windows in the third subject, this will be needed to calculate the last index
            n_windows1 = self.id_size_dict[sub_id1]
            n_windows2 = self.id_size_dict[sub_id2]
            n_windows3 = self.id_size_dict[sub_id3]

            first_index1 = self.index_map_inverse[(sub_id1, 0)]
            last_index1 = self.index_map_inverse[(sub_id1, n_windows1 - 1)]

            first_index2 = self.index_map_inverse[(sub_id2, 0)]
            last_index2 = self.index_map_inverse[(sub_id2, n_windows2 - 1)]

            first_index3 = self.index_map_inverse[(sub_id3, 0)]
            last_index3 = self.index_map_inverse[(sub_id3, n_windows3 - 1)]

            indices1 = [i for i in range(first_index1, last_index1 + 1) if i not in used_1d_indices]
            indices2 = [i for i in range(first_index2, last_index2 + 1) if i not in used_1d_indices]
            indices3 = [i for i in range(first_index3, last_index3 + 1) if i not in used_1d_indices]

            combined_indices = [*indices1, *indices2, *indices3]
            n_combined_indices = len(combined_indices)

            # Check if the windows available for sampling are enough for a batch:
            while n_combined_indices < self.batch_size:
                # If three subjects do not have enough windows then use more:
                next_sub_id = next(pool)

                # Get the number of windows in the third subject, this will be needed to calculate the last index
                n_windows = self.id_size_dict[next_sub_id]

                first_index = self.index_map_inverse[(next_sub_id, 0)]
                last_index = self.index_map_inverse[(next_sub_id, n_windows - 1)]
                indices_to_add = [i for i in range(first_index, last_index + 1) if i not in used_1d_indices]
                combined_indices.extend(indices_to_add)

            batch_indices = self.rng.sample(combined_indices, self.batch_size)
            used_1d_indices.update(batch_indices)
            yield batch_indices
            batches += 1

        # The last batch may contain less than the batch_size:
        unused_1d_indices = []
        for sub_id in self.subject_ids:
            n_windows = self.id_size_dict[sub_id]
            for window_index in range(n_windows):
                index_1d = self.index_map_inverse[(sub_id, window_index)]
                if index_1d not in used_1d_indices:
                    unused_1d_indices.append(index_1d)

        assert len(unused_1d_indices) <= self.batch_size
        yield unused_1d_indices


def get_n_train_windows_by_id(subject_id: int) -> int:
    sub_dir = EXPANDED_ARRAYS_DIR.joinpath(str(subject_id).zfill(4), "train")
    dir_iter = sub_dir.iterdir()
    n_windows = 0
    for fpath in dir_iter:
        if fpath.is_file():
            n_windows += 1
    return n_windows


def get_n_test_windows_by_id(subject_id: int) -> int:
    sub_dir = EXPANDED_ARRAYS_DIR.joinpath(str(subject_id).zfill(4), "test")
    dir_iter = sub_dir.iterdir()
    n_windows = 0
    for fpath in dir_iter:
        if fpath.is_file():
            n_windows += 1
    return n_windows


def train_pleth_window_loader(sub_id: int, window_id: int) -> tuple[np.array, np.array]:
    X_path = EXPANDED_ARRAYS_DIR.joinpath(str(sub_id).zfill(4), "train", f"X_{window_id}.npy")
    y_path = EXPANDED_ARRAYS_DIR.joinpath(str(sub_id).zfill(4), "train", f"y_{window_id}.npy")
    X = np.load(X_path).reshape(-1, WINDOW_SAMPLES_SIZE, N_SIGNALS).astype("float32")

    # Drop the flow signal since only Pleth will be used as input:
    X = np.delete(X, 0, axis=2)

    y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")
    return X.ravel(), y.ravel()


def test_pleth_window_loader(sub_id: int, window_id: int) -> tuple[np.array, np.array]:
    X_path = EXPANDED_ARRAYS_DIR.joinpath(str(sub_id).zfill(4), "test", f"X_{window_id}.npy")
    y_path = EXPANDED_ARRAYS_DIR.joinpath(str(sub_id).zfill(4), "test", f"y_{window_id}.npy")
    X = np.load(X_path).reshape(-1, WINDOW_SAMPLES_SIZE, N_SIGNALS).astype("float32")

    # Drop the flow signal since only Pleth will be used as input:
    X = np.delete(X, 0, axis=2)

    y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")
    return X.ravel(), y.ravel()


def get_saved_pleth_train_dataloader() -> DataLoader:
    train_loader_object_path = Path(PATH_TO_SUBSET1).joinpath("TrainLoaderObj.pickle")
    if train_loader_object_path.is_file():
        with open(train_loader_object_path, "rb") as f:
            return pickle.load(f)


def get_saved_pleth_test_dataloader() -> DataLoader:
    test_loader_object_path = Path(PATH_TO_SUBSET1).joinpath("TestLoaderObj.pickle")
    if test_loader_object_path.is_file():
        with open(test_loader_object_path, "rb") as f:
            return pickle.load(f)


def get_saved_pleth_test_cross_sub_dataloader() -> DataLoader:
    ptest_cross_sub_loader_object_path = Path(PATH_TO_SUBSET1).joinpath("TestCrossSubLoaderObj.pickle")
    if ptest_cross_sub_loader_object_path.is_file():
        with open(ptest_cross_sub_loader_object_path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":

    # train_ids_file = Path(PATH_TO_SUBSET1).joinpath("trainIds.npy")
    # test_ids_file = Path(PATH_TO_SUBSET1).joinpath("testIds.npy")
    # if train_ids_file.is_file() and test_ids_file.is_file():
    #     train_ids = list(np.load(train_ids_file))
    #     test_ids = list(np.load(test_ids_file))
    # else:
    # Get all ids in the directory with arrays. Each subdir is one subject
    subset_ids = [int(f.name) for f in EXPANDED_ARRAYS_DIR.iterdir() if f.is_dir()]

    random.seed(33)
    test_ids = random.sample(subset_ids, 2)
    train_ids = [id for id in subset_ids if id not in test_ids]

    # np.save(train_ids_file, np.array(train_ids).astype("uint8"))
    # np.save(test_ids_file, np.array(test_ids).astype("uint8"))

    print(test_ids)
    print(train_ids)

    # Try to load previously created train loader:
    train_loader_object_file = Path(PATH_TO_SUBSET1).joinpath("TrainLoaderObj.pickle")
    test_loader_object_file = Path(PATH_TO_SUBSET1).joinpath("TestLoaderObj.pickle")
    test_cross_sub_loader_object_file = Path(PATH_TO_SUBSET1).joinpath("TestCrossSubLoaderObj.pickle")
    if train_loader_object_file.is_file() and test_loader_object_file.is_file() and test_cross_sub_loader_object_file.is_file():
        pleth_train_loader = get_saved_pleth_train_dataloader()
        pleth_test_loader = get_saved_pleth_test_dataloader()
        pleth_test_cross_sub_loader = get_saved_pleth_test_cross_sub_dataloader()
    else:
        train_set = Dataset(subject_ids=train_ids, window_loader=train_pleth_window_loader,
                            get_n_windows_by_id=get_n_train_windows_by_id,
                            transform=torch.from_numpy, target_transform=torch.from_numpy)
        test_set = Dataset(subject_ids=train_ids, window_loader=test_pleth_window_loader,
                           get_n_windows_by_id=get_n_test_windows_by_id,
                           transform=torch.from_numpy, target_transform=torch.from_numpy)
        test_cross_sub_set = Dataset(subject_ids=test_ids, window_loader=test_pleth_window_loader,
                                     get_n_windows_by_id=get_n_test_windows_by_id,
                                     transform=torch.from_numpy, target_transform=torch.from_numpy)

        train_sampler = BatchSampler(train_ids, batch_size=128,
                                     index_map=train_set.index_map,
                                     id_size_dict=train_set.id_size_dict,
                                     shuffle=True,
                                     seed=33)
        test_sampler = BatchSampler(train_ids, batch_size=128,
                                    index_map=test_set.index_map,
                                    id_size_dict=test_set.id_size_dict,
                                    shuffle=True,
                                    seed=33)
        test_cross_sub_sampler = BatchSampler(test_ids, batch_size=128,
                                              index_map=test_cross_sub_set.index_map,
                                              id_size_dict=test_cross_sub_set.id_size_dict,
                                              shuffle=True,
                                              seed=33)

        print(f"Train sampler length: {len(train_sampler)}")
        print(f"Test sampler length: {len(test_sampler)}")
        print(f"Test_cross sampler length: {len(test_cross_sub_sampler)}")

        pleth_train_loader = DataLoader(train_set, shuffle=False, batch_sampler=train_sampler)
        pleth_test_loader = DataLoader(test_set, shuffle=False, batch_sampler=test_sampler)
        pleth_test_cross_sub_loader = DataLoader(test_cross_sub_set, shuffle=False, batch_sampler=test_cross_sub_sampler)

        # # Save train loader for future use
        # with open(train_loader_object_file, "wb") as file:
        #     pickle.dump(train_loader, file)
        #
        # # Save test loader for future use
        # with open(test_loader_object_file, "wb") as file:
        #     pickle.dump(test_loader, file)
        #
        # # Save test cross subject loader for future use
        # with open(test_cross_sub_loader_object_file, "wb") as file:
        #     pickle.dump(test_cross_sub_loader, file)

    print(f"Batches in epoch: {len(pleth_test_loader)}")

    test_iter = iter(pleth_test_loader)
    for (i, item) in enumerate(test_iter):
        X, y = item
        print(f"batch: {i},  X shape: {X.shape},  y shape: {y.shape}")
        # print(X.dtype)
        # print(y)
        # memory_usage_in_bytes = X.element_size() * X.nelement() + y.element_size() * y.nelement()
        # print(f"Memory Usage: {memory_usage_in_bytes} bytes")

