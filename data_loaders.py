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
    PATH_TO_SUBSET1 = config["paths"]["local"]["subset_1_directory"]
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")

ARRAYS_DIR = Path(PATH_TO_SUBSET1).joinpath("arrays")


class PlethToLabelDataset(Dataset):

    def __init__(self, subject_ids: list[int], arrays_loader: Callable[[int], tuple[np.array, np.array]],
                 transform=None, target_transform=None):
        self.subject_ids = subject_ids
        self.load_arrays = arrays_loader
        self.transform = transform
        self.target_transform = target_transform

        # Since the inputs are indexed with two numbers (id, window_index),
        # we need a mapping from one dimensional index to two-dimensional index:
        self.index_map = {}
        self.id_size_dict = {}
        index = 0
        for id in self.subject_ids:
            _, y = self.load_arrays(id)
            n_windows = y.shape[0]
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
        X, y = self.load_arrays(subject_id)

        # Drop the flow signal since only Pleth will be used as input:
        X = np.delete(X, 0, axis=2)

        signal = X[window_id, :].ravel()
        labels = y[window_id, :].ravel()
        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            labels = self.target_transform(labels)
        return signal, labels


class BatchSampler(Sampler[list[int]]):
    def __init__(self, subject_ids: list[int], batch_size: int,
                 arrays_loader: Callable[[int], tuple[np.array, np.array]],
                 index_map: dict[int: tuple[int, int]],
                 id_size_dict: dict[int: int] | None,
                 shuffle=False,
                 seed=None) -> None:
        """
        :param subject_ids:
        :param batch_size:
        :param arrays_loader: Functions that loads the X, y arrays of a subject
        :param index_map: Index map provided by the Dataset
        :param id_size_dict:
        :param shuffle: Whether to shuffle the subjects between epochs
        :param seed: The seed to use for shuffling and sampling
        """

        self.subject_ids = subject_ids
        self.index_map = index_map
        # Calculate inverse index_map, which maps (subject_id, window_id) to id
        self.index_map_inverse = {v: k for k, v in index_map.items()}

        # Save new array loader function for specific array dir:
        self.load_arrays = arrays_loader
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.rng = random.Random()
        if seed is not None:
            self.rng = random.Random(seed)

        if id_size_dict is None:
            self.id_size_dict = {}
        else:
            self.id_size_dict = id_size_dict

    def __len__(self) -> int:
        # Number of batches:
        return (len(self.index_map) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # for batch in torch.chunk(torch.argsort(sizes), len(self)):
        #     yield batch.tolist()

        batches = 0
        used_indices = SortedList()

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
            if sub_id3 in self.id_size_dict:
                n_windows1 = self.id_size_dict[sub_id1]
                n_windows2 = self.id_size_dict[sub_id2]
                n_windows3 = self.id_size_dict[sub_id3]
            else:
                _, y1 = self.load_arrays(sub_id1)
                _, y2 = self.load_arrays(sub_id2)
                _, y3 = self.load_arrays(sub_id3)
                n_windows1 = y1.shape[0]
                n_windows2 = y2.shape[0]
                n_windows3 = y3.shape[0]
                self.id_size_dict[sub_id1] = n_windows1
                self.id_size_dict[sub_id2] = n_windows2
                self.id_size_dict[sub_id3] = n_windows3

            first_index1 = self.index_map_inverse[(sub_id1, 0)]
            last_index1 = self.index_map_inverse[(sub_id1, n_windows1 - 1)]

            first_index2 = self.index_map_inverse[(sub_id2, 0)]
            last_index2 = self.index_map_inverse[(sub_id2, n_windows2 - 1)]

            first_index3 = self.index_map_inverse[(sub_id3, 0)]
            last_index3 = self.index_map_inverse[(sub_id3, n_windows3 - 1)]

            indices1 = [i for i in range(first_index1, last_index1 + 1) if i not in used_indices]
            indices2 = [i for i in range(first_index2, last_index2 + 1) if i not in used_indices]
            indices3 = [i for i in range(first_index3, last_index3 + 1) if i not in used_indices]

            combined_indices = [*indices1, *indices2, *indices3]
            n_combined_indices = len(combined_indices)

            # Check if the windows available for sampling are enough for a batch:
            while n_combined_indices < self.batch_size:
                # If three subjects do not have enough windows then use more:
                next_sub_id = next(pool)

                # Get the number of windows in the third subject, this will be needed to calculate the last index
                if next_sub_id in self.id_size_dict:
                    n_windows = self.id_size_dict[next_sub_id]
                else:
                    _, y = self.load_arrays(next_sub_id)
                    n_windows = y.shape[0]
                    self.id_size_dict[next_sub_id] = n_windows

                first_index = self.index_map_inverse[(next_sub_id, 0)]
                last_index = self.index_map_inverse[(next_sub_id, n_windows - 1)]
                indices_to_add = [i for i in range(first_index, last_index + 1) if i not in used_indices]
                combined_indices.extend(indices_to_add)

            batch_indices = self.rng.sample(combined_indices, self.batch_size)
            used_indices.update(batch_indices)
            yield batch_indices
            batches += 1

        # The last batch may contain less than the batch_size:
        unused_indices = [i for i in self.index_map if i not in used_indices]
        assert len(unused_indices) <= self.batch_size
        yield unused_indices


def get_subject_train_arrays(subject_arrays_dir: Path, subject_id: int) -> tuple[np.array, np.array]:
    X_path = subject_arrays_dir.joinpath(str(subject_id).zfill(4)).joinpath("X_train.npy")
    y_path = subject_arrays_dir.joinpath(str(subject_id).zfill(4)).joinpath("y_train.npy")
    X = np.load(X_path).reshape(-1, WINDOW_SAMPLES_SIZE, N_SIGNALS).astype("float32")
    y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")
    return X, y


def get_subject_test_arrays(subject_arrays_dir: Path, subject_id: int) -> tuple[np.array, np.array]:
    X_path = subject_arrays_dir.joinpath(str(subject_id).zfill(4)).joinpath("X_test.npy")
    y_path = subject_arrays_dir.joinpath(str(subject_id).zfill(4)).joinpath("y_test.npy")
    X = np.load(X_path).reshape(-1, WINDOW_SAMPLES_SIZE, N_SIGNALS).astype("float32")
    y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")
    return X, y


def train_array_loader(sub_id: int) -> tuple[np.array, np.array]:
    return get_subject_train_arrays(ARRAYS_DIR, sub_id)


def test_array_loader(sub_id: int) -> tuple[np.array, np.array]:
    return get_subject_test_arrays(ARRAYS_DIR, sub_id)


def get_saved_train_loader() -> DataLoader:
    train_loader_object_path = Path(PATH_TO_SUBSET1).joinpath("TrainLoaderObj.pickle")
    if train_loader_object_path.is_file():
        with open(train_loader_object_path, "rb") as f:
            return pickle.load(f)


def get_saved_test_loader() -> DataLoader:
    test_loader_object_path = Path(PATH_TO_SUBSET1).joinpath("TestLoaderObj.pickle")
    if test_loader_object_path.is_file():
        with open(test_loader_object_path, "rb") as f:
            return pickle.load(f)


def get_saved_test_cross_sub_loader() -> DataLoader:
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
    subset_ids = [int(f.name) for f in ARRAYS_DIR.iterdir() if f.is_dir()]

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
        train_loader = get_saved_train_loader()
        test_loader = get_saved_test_loader()
        test_cross_sub_loader = get_saved_test_cross_sub_loader()
    else:
        train_set = PlethToLabelDataset(subject_ids=train_ids, arrays_loader=train_array_loader,
                                        transform=torch.from_numpy, target_transform=torch.from_numpy)
        test_set = PlethToLabelDataset(subject_ids=train_ids, arrays_loader=test_array_loader,
                                       transform=torch.from_numpy, target_transform=torch.from_numpy)
        test_cross_sub_set = PlethToLabelDataset(subject_ids=test_ids, arrays_loader=test_array_loader,
                                                 transform=torch.from_numpy, target_transform=torch.from_numpy)

        train_sampler = BatchSampler(train_ids, batch_size=128, arrays_loader=train_array_loader,
                                     index_map=train_set.index_map,
                                     id_size_dict=train_set.id_size_dict,
                                     shuffle=True,
                                     seed=33)
        test_sampler = BatchSampler(train_ids, batch_size=128, arrays_loader=test_array_loader,
                                    index_map=test_set.index_map,
                                    id_size_dict=test_set.id_size_dict,
                                    shuffle=True,
                                    seed=33)
        test_cross_sub_sampler = BatchSampler(test_ids, batch_size=128, arrays_loader=test_array_loader,
                                              index_map=test_cross_sub_set.index_map,
                                              id_size_dict=test_cross_sub_set.id_size_dict,
                                              shuffle=True,
                                              seed=33)
        train_loader = DataLoader(train_set, shuffle=False, batch_sampler=train_sampler)
        test_loader = DataLoader(test_set, shuffle=False, batch_sampler=test_sampler)
        test_cross_sub_loader = DataLoader(test_cross_sub_set, shuffle=False, batch_sampler=test_cross_sub_sampler)

        # Save train loader for future use
        with open(train_loader_object_file, "wb") as file:
            pickle.dump(train_loader, file)

        # Save test loader for future use
        with open(test_loader_object_file, "wb") as file:
            pickle.dump(test_loader, file)

        # Save test cross subject loader for future use
        with open(test_cross_sub_loader_object_file, "wb") as file:
            pickle.dump(test_cross_sub_loader, file)

    print(f"Batches in epoch: {len(test_loader)}")

    test_iter = iter(test_loader)
    for (i, item) in enumerate(test_iter):
        X, y = item
        print(f"batch: {i},  X shape: {X.shape},  y shape: {y.shape}")
        # print(X.dtype)
        # print(y)
        # memory_usage_in_bytes = X.element_size() * X.nelement() + y.element_size() * y.nelement()
        # print(f"Memory Usage: {memory_usage_in_bytes} bytes")

