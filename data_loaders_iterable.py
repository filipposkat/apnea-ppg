from itertools import cycle
from pathlib import Path
from typing import Callable

import yaml

import numpy as np
import random
import pickle
from sortedcontainers import SortedList

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

GENERATE_TRAIN_TEST_SPLIT = True
WINDOW_SAMPLES_SIZE = 512
N_SIGNALS = 2
CROSS_SUBJECT_TEST_SIZE = 100
BATCH_WINDOW_SAMPLING_RATIO = 0.1
BATCH_SIZE = 256
INCLUDE_TRAIN_IN_CROSS_SUB_TESTING = False

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1

ARRAYS_DIR = PATH_TO_SUBSET1.joinpath("arrays")

# Paths for saving dataloaders:
dataloaders_path = PATH_TO_SUBSET1_TRAINING.joinpath("dataloaders")


# Get all ids in the directory with arrays. Each subdir is one subject
if GENERATE_TRAIN_TEST_SPLIT:
    subset_ids = [int(f.name) for f in ARRAYS_DIR.iterdir() if f.is_dir()]
    rng = random.Random(33)
    test_ids = rng.sample(subset_ids, 2)
    train_ids = [id for id in subset_ids if id not in test_ids]
else:
    test_ids = [1266, 939]
    train_ids = [1017, 196, 892, 713, 1573, 620, 934, 468, 718, 589, 183, 33, 1453, 435, 505, 1281, 675, 405, 935, 133,
                 27, 1604, 1464, 1089, 658, 332, 719, 1087, 527, 679, 643, 1236, 1278, 796, 979, 1650, 1133, 140, 626,
                 381, 1478, 571, 743, 1376, 407, 1623, 628, 64, 1342, 951, 931, 1626, 1271, 715, 1291, 155, 728, 1212,
                 1501, 1161, 1010, 1656, 1019, 194, 811, 823, 744, 912, 1013, 1502, 712, 1128, 1263, 490, 220, 1589,
                 1552, 107, 303, 937, 346, 386, 725, 1562, 1016, 1328, 917, 1224, 860, 1497, 1301, 561, 125, 651, 572,
                 863, 648, 1356]


class IterDataset(IterableDataset):
    load_arrays: Callable[[int], tuple[np.array, np.array]]

    def __init__(self, subject_ids: list[int],
                 batch_size: int,
                 dataset_split_type: str = "train",
                 desired_target: str = "pleth",
                 shuffle=True,
                 seed=33,
                 transform=None,
                 target_transform=None) -> None:
        """
        :param subject_ids:
        :param batch_size:
        :param dataset_split_type: One of: train, test or cross_test
        :param desired_target: One of: label, flow
        :param shuffle: Whether to shuffle the subjects between epochs
        :param seed: The seed to use for shuffling and sampling
        """
        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.target = desired_target

        if dataset_split_type == "train":
            self.load_arrays = self.train_array_loader
        elif dataset_split_type == "test":
            self.load_arrays = self.test_array_loader
        else:
            self.load_arrays = self.cross_test_array_loader

        self.transform = transform
        self.target_transform = target_transform

        # Determine n_signals / channels (if pleth only =1 if pleth and flow then =2):
        X, _ = self.load_arrays(self.subject_ids[0])
        n_signals = X.shape[1]
        assert X.shape[0] > batch_size
        assert n_signals == 1  # Flow has been deleted already
        assert X.shape[2] == WINDOW_SAMPLES_SIZE

        # Inputs are indexed with two numbers (id, window_index),
        self.id_size_dict = {}
        self.total_windows = 0
        for id in self.subject_ids:
            _, y = self.load_arrays(id)
            n_windows = y.shape[0]
            self.id_size_dict[id] = n_windows
            self.total_windows += n_windows

        self.shuffle = shuffle
        self.rng = random.Random()
        if seed is not None:
            self.rng = random.Random(seed)

    def __len__(self) -> int:
        # Number of batches:
        return (self.total_windows + self.batch_size - 1) // self.batch_size

    def train_array_loader(self, sub_id: int) -> tuple[np.array, np.array]:
        X_path = ARRAYS_DIR.joinpath(str(sub_id).zfill(4)).joinpath("X_train.npy")
        y_path = ARRAYS_DIR.joinpath(str(sub_id).zfill(4)).joinpath("y_train.npy")

        X = np.load(X_path).astype("float32")  # shape: (n_windows, window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=1, axis2=2)  # shape: (n_windows, n_signals, window_size), Flow comes first

        if "flow" == self.target:
            y = X[:, 0, :]
        else:
            y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")

        # Drop the flow signal since only Pleth will be used as input:
        X = np.delete(X, 0, axis=1)
        return X, y

    def test_array_loader(self, sub_id: int) -> tuple[np.array, np.array]:
        X_path = ARRAYS_DIR.joinpath(str(sub_id).zfill(4)).joinpath("X_test.npy")
        y_path = ARRAYS_DIR.joinpath(str(sub_id).zfill(4)).joinpath("y_test.npy")

        X = np.load(X_path).astype("float32")  # shape: (n_windows, window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=1, axis2=2)  # shape: (n_windows, n_signals, window_size), Flow comes first

        if "flow" == self.target:
            y = X[:, 0, :]
        else:
            y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")

        # Drop the flow signal since only Pleth will be used as input:
        X = np.delete(X, 0, axis=1)
        return X, y

    def cross_test_array_loader(self, sub_id: int) -> tuple[np.array, np.array]:
        X_test, y_test = self.test_array_loader(sub_id)
        if INCLUDE_TRAIN_IN_CROSS_SUB_TESTING:
            X_train, y_train = self.train_array_loader(sub_id)
            X = np.concatenate((X_train, X_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)
        else:
            X = X_test
            y = y_test
        return X, y

    def get_specific_windows(self, subject_id: int, windows_indices: list[int]) -> tuple[np.array, np.array]:
        X, y = self.load_arrays(subject_id)

        signals = X[windows_indices, :]
        labels = y[windows_indices, :]
        return signals, labels

    def __iter__(self):
        # for batch in torch.chunk(torch.argsort(sizes), len(self)):
        #     yield batch.tolist()

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

            # Get sampled arrays:
            X_batch = []
            y_batch = []
            for sub_id in ids_tmp:
                windows_indices = [two_dim_index[1] for two_dim_index in batch_2d_indices if two_dim_index[0] == sub_id]
                signals, labels = self.get_specific_windows(sub_id, windows_indices)
                X_batch.append(signals)
                y_batch.append(labels)

            X_batch = np.concatenate(X_batch, axis=0)
            y_batch = np.concatenate(y_batch, axis=0)

            if self.transform:
                X_batch = self.transform(X_batch)
            if self.target_transform:
                y_batch = self.target_transform(y_batch)
            yield X_batch, y_batch

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

        remaining_subs = [sub_window[0] for sub_window in unused_2d_indices]
        assert len(unused_2d_indices) <= self.batch_size

        # Get sampled arrays:
        X_batch = []
        y_batch = []
        for sub_id in remaining_subs:
            windows_indices = [two_dim_index[1] for two_dim_index in unused_2d_indices if two_dim_index[0] == sub_id]
            signals, labels = self.get_specific_windows(sub_id, windows_indices)
            X_batch.append(signals)
            y_batch.append(labels)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        if self.transform:
            X_batch = self.transform(X_batch)
        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        yield X_batch, y_batch


def get_saved_train_loader(batch_size=BATCH_SIZE) -> DataLoader:
    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Iterable_Train_Loader.pickle")
    if not object_file.exists():
        loader = get_new_train_loader(batch_size=batch_size)
        # Save train loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

        return loader

    if object_file.is_file():
        with open(object_file, "rb") as f:
            return pickle.load(f)


def get_new_train_loader(batch_size=BATCH_SIZE) -> DataLoader:
    train_set = IterDataset(subject_ids=train_ids,
                            batch_size=batch_size,
                            dataset_split_type="train",
                            shuffle=True,
                            seed=33,
                            transform=torch.from_numpy,
                            target_transform=torch.from_numpy)
    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    return DataLoader(train_set, batch_size=None, pin_memory=True)


def get_saved_test_loader(batch_size=BATCH_SIZE) -> DataLoader:
    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Iterable_Test_Loader.pickle")
    if not object_file.exists():
        loader = get_new_test_loader(batch_size=batch_size)
        # Save train loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)
        return loader

    if object_file.is_file():
        with open(object_file, "rb") as f:
            return pickle.load(f)


def get_new_test_loader(batch_size=BATCH_SIZE) -> DataLoader:
    test_set = IterDataset(subject_ids=train_ids,
                           batch_size=batch_size,
                           dataset_split_type="test",
                           shuffle=True,
                           seed=33,
                           transform=torch.from_numpy,
                           target_transform=torch.from_numpy)
    return DataLoader(test_set, batch_size=None, pin_memory=True)


def get_saved_test_cross_sub_loader(batch_size=BATCH_SIZE) -> DataLoader:
    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Iterable_TestCrossSub_Loader.pickle")
    if not object_file.exists():
        loader = get_new_test_cross_sub_loader(batch_size=batch_size)
        # Save train loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)
        return loader

    if object_file.is_file():
        with open(object_file, "rb") as f:
            return pickle.load(f)


def get_new_test_cross_sub_loader(batch_size=BATCH_SIZE) -> DataLoader:
    test_cross_sub_set = IterDataset(subject_ids=test_ids,
                                     batch_size=batch_size,
                                     dataset_split_type="cross_test",
                                     shuffle=True,
                                     seed=33,
                                     transform=torch.from_numpy,
                                     target_transform=torch.from_numpy)
    return DataLoader(test_cross_sub_set, batch_size=None, pin_memory=True)


if __name__ == "__main__":

    # id = 107
    # print(id)
    # X_path = ARRAYS_DIR.joinpath(str(id).zfill(4)).joinpath("X_train.npy")
    # y_path = ARRAYS_DIR.joinpath(str(id).zfill(4)).joinpath("y_train.npy")
    # X = np.load(X_path)
    # y = np.load(y_path)
    # import scipy.io as io
    # io.savemat('107.mat', dict(x=X[0:10, :, :], y=y[0:10, :]))
    #
    # X_train, y_train = train_array_loader(sub_id=id)
    # print(X_train[0,0,:])
    print(test_ids)
    print(train_ids)

    train_set = IterDataset(subject_ids=train_ids,
                            batch_size=BATCH_SIZE,
                            dataset_split_type="train",
                            shuffle=True,
                            seed=33,
                            transform=torch.from_numpy,
                            target_transform=torch.from_numpy)
    test_set = IterDataset(subject_ids=train_ids,
                           batch_size=BATCH_SIZE,
                           dataset_split_type="test",
                           shuffle=True,
                           seed=33,
                           transform=torch.from_numpy,
                           target_transform=torch.from_numpy)
    test_cross_sub_set = IterDataset(subject_ids=test_ids,
                                     batch_size=BATCH_SIZE,
                                     dataset_split_type="cross_test",
                                     shuffle=True,
                                     seed=33,
                                     transform=torch.from_numpy,
                                     target_transform=torch.from_numpy)

    print(f"Train batches: {len(train_set)}")
    print(f"Test batches: {len(test_set)}")
    print(f"Test_cross batches: {len(test_cross_sub_set)}")

    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    train_loader = DataLoader(train_set, batch_size=None, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=None, pin_memory=True)
    test_cross_sub_loader = DataLoader(test_cross_sub_set, batch_size=None, pin_memory=True)

    dataloaders_path.joinpath(f"bs{BATCH_SIZE}").mkdir(parents=True, exist_ok=True)
    train_loader_object_file = dataloaders_path.joinpath(f"bs{BATCH_SIZE}",
                                                         f"PlethToLabel_Iterable_Train_Loader.pickle")
    test_loader_object_file = dataloaders_path.joinpath(f"bs{BATCH_SIZE}",
                                                        f"PlethToLabel_Iterable_Test_Loader.pickle")
    test_cross_sub_loader_object_file = dataloaders_path.joinpath(f"bs{BATCH_SIZE}",
                                                                  f"PlethToLabel_Iterable_TestCrossSub_Loader.pickle")

    # Save train loader for future use
    with open(train_loader_object_file, "wb") as file:
        pickle.dump(train_loader, file)

    # Save test loader for future use
    with open(test_loader_object_file, "wb") as file:
        pickle.dump(test_loader, file)

    # Save test cross subject loader for future use
    with open(test_cross_sub_loader_object_file, "wb") as file:
        pickle.dump(test_cross_sub_loader, file)

    loader = train_loader
    batches = len(loader)
    print(f"Batches in epoch: {batches}")
    iter = iter(loader)
    for (i, item) in tqdm(enumerate(iter), total=batches):
        X, y = item
        print(f"batch: {i}/{batches},  X shape: {X.shape},  y shape: {y.shape}")
        # print(X.dtype)
        # print(y)
        # memory_usage_in_bytes = X.element_size() * X.nelement() + y.element_size() * y.nelement()
        # print(f"Memory Usage: {memory_usage_in_bytes} bytes")
