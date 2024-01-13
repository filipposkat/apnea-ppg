from itertools import cycle
from pathlib import Path
from typing import Callable
import math

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
SEED = 33
NUM_WORKERS = 2  # better to use power of two, otherwise each worker will have different number of subject ids
PREFETCH_FACTOR = 2
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
    cross_sub_test_ids = rng.sample(subset_ids, CROSS_SUBJECT_TEST_SIZE)
    train_ids = [id for id in subset_ids if id not in cross_sub_test_ids]
else:
    cross_sub_test_ids = [5002, 1453, 5396, 2030, 2394, 4047, 5582, 4478, 4437, 1604, 6726, 5311, 4229, 2780, 5957,
                          6697, 4057, 3823, 2421, 5801, 5451, 679, 2636, 3556, 2688, 4322, 4174, 572, 5261, 5847, 3671,
                          2408, 2771, 4671, 5907, 2147, 979, 620, 6215, 2434, 1863, 651, 3043, 1016, 5608, 6538, 2126,
                          4270, 2374, 6075, 107, 3013, 4341, 5695, 2651, 6193, 3332, 3314, 1589, 935, 386, 3042, 5393,
                          4794, 6037, 648, 1271, 811, 1010, 2750, 33, 626, 3469, 6756, 2961, 1756, 1650, 3294, 3913,
                          5182, 4014, 3025, 5148, 4508, 3876, 2685, 4088, 675, 125, 6485, 3239, 5231, 3037, 5714, 5986,
                          155, 4515, 6424, 2747, 1356]
    train_ids = [27, 64, 133, 140, 183, 194, 196, 220, 303, 332, 346, 381, 405, 407, 435, 468, 490, 505, 527, 561, 571,
                 589, 628, 643, 658, 712, 713, 715, 718, 719, 725, 728, 743, 744, 796, 823, 860, 863, 892, 912, 917,
                 931, 934, 937, 939, 951, 1013, 1017, 1019, 1087, 1089, 1128, 1133, 1161, 1212, 1224, 1236, 1263, 1266,
                 1278, 1281, 1291, 1301, 1328, 1342, 1376, 1464, 1478, 1497, 1501, 1502, 1552, 1562, 1573, 1623, 1626,
                 1656, 1693, 1733, 1738, 1790, 1797, 1809, 1833, 1838, 1874, 1879, 1906, 1913, 1914, 1924, 1983, 2003,
                 2024, 2039, 2105, 2106, 2118, 2204, 2208, 2216, 2227, 2239, 2246, 2251, 2264, 2276, 2291, 2292, 2317,
                 2345, 2375, 2397, 2451, 2452, 2467, 2468, 2523, 2539, 2572, 2614, 2665, 2701, 2735, 2781, 2798, 2800,
                 2802, 2819, 2834, 2848, 2877, 2879, 2881, 2897, 2915, 2934, 2995, 3012, 3024, 3028, 3106, 3149, 3156,
                 3204, 3223, 3236, 3275, 3280, 3293, 3324, 3337, 3347, 3352, 3419, 3439, 3452, 3468, 3555, 3564, 3575,
                 3591, 3603, 3604, 3652, 3690, 3702, 3711, 3734, 3743, 3770, 3781, 3803, 3833, 3852, 3854, 3867, 3902,
                 3933, 3934, 3967, 3974, 3980, 3987, 3992, 4029, 4038, 4085, 4099, 4123, 4128, 4157, 4163, 4205, 4228,
                 4250, 4252, 4254, 4256, 4295, 4296, 4330, 4332, 4428, 4462, 4496, 4497, 4511, 4541, 4544, 4554, 4592,
                 4624, 4661, 4734, 4820, 4826, 4878, 4912, 4948, 5029, 5053, 5063, 5075, 5096, 5101, 5118, 5137, 5155,
                 5162, 5163, 5179, 5203, 5214, 5232, 5276, 5283, 5308, 5339, 5357, 5358, 5365, 5387, 5395, 5433, 5457,
                 5472, 5480, 5491, 5503, 5565, 5580, 5662, 5686, 5697, 5703, 5753, 5788, 5798, 5845, 5897, 5909, 5954,
                 5982, 6009, 6022, 6047, 6050, 6052, 6074, 6077, 6117, 6174, 6180, 6244, 6261, 6274, 6279, 6280, 6291,
                 6316, 6318, 6322, 6351, 6366, 6390, 6417, 6422, 6492, 6528, 6549, 6616, 6682, 6695, 6704, 6755, 6781,
                 6804, 6807, 6811]


class IterDataset(IterableDataset):
    load_arrays: Callable[[int], tuple[np.array, np.array]]
    arrays_dir: Path

    def __init__(self, subject_ids: list[int],
                 batch_size: int,
                 dataset_split_type: str = "train",
                 desired_target: str = "pleth",
                 shuffle=True,
                 seed=33,
                 transform=None,
                 target_transform=None,
                 initialization_progress=True) -> None:
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
        self.arrays_dir = ARRAYS_DIR

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

        subject_ids = tqdm(self.subject_ids) if initialization_progress else self.subject_ids
        for id in subject_ids:
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
        X_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("X_train.npy")
        y_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("y_train.npy")

        X = np.load(str(X_path)).astype("float32")  # shape: (n_windows, window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=1, axis2=2)  # shape: (n_windows, n_signals, window_size), Flow comes first

        if "flow" == self.target:
            y = X[:, 0, :]
        else:
            y = np.load(str(y_path)).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")

        # Drop the flow signal since only Pleth will be used as input:
        X = np.delete(X, 0, axis=1)
        return X, y

    def test_array_loader(self, sub_id: int) -> tuple[np.array, np.array]:
        X_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("X_test.npy")
        y_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("y_test.npy")

        X = np.load(str(X_path)).astype("float32")  # shape: (n_windows, window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=1, axis2=2)  # shape: (n_windows, n_signals, window_size), Flow comes first

        if "flow" == self.target:
            y = X[:, 0, :]
        else:
            y = np.load(str(y_path)).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")

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


def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    dataset: IterDataset = worker_info.dataset  # the dataset copy in this worker process
    sub_ids = dataset.subject_ids

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil(len(sub_ids) / float(num_workers)))
    start = worker_id * per_worker
    end = min(start + per_worker, len(sub_ids))

    dataset.subject_ids = sub_ids[start:end]


def get_new_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR) -> DataLoader:
    train_set = IterDataset(subject_ids=train_ids,
                            batch_size=batch_size,
                            dataset_split_type="train",
                            shuffle=True,
                            seed=SEED,
                            transform=torch.from_numpy,
                            target_transform=torch.from_numpy)
    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    return DataLoader(train_set, batch_size=None, pin_memory=True,
                      num_workers=num_workers, worker_init_fn=worker_init_fn, prefetch_factor=pre_fetch)


def get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR, arrays_dir: Path = None) \
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
        loader.worker_init_fn = worker_init_fn
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

    return loader


def get_new_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR) -> DataLoader:
    test_set = IterDataset(subject_ids=train_ids,
                           batch_size=batch_size,
                           dataset_split_type="test",
                           shuffle=True,
                           seed=SEED,
                           transform=torch.from_numpy,
                           target_transform=torch.from_numpy)
    return DataLoader(test_set, batch_size=None, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch, worker_init_fn=worker_init_fn)


def get_saved_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR, arrays_dir: Path = None) \
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
        loader.worker_init_fn = worker_init_fn
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

    return loader


def get_new_test_cross_sub_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR) -> DataLoader:
    test_cross_sub_set = IterDataset(subject_ids=cross_sub_test_ids,
                                     batch_size=batch_size,
                                     dataset_split_type="cross_test",
                                     shuffle=True,
                                     seed=SEED,
                                     transform=torch.from_numpy,
                                     target_transform=torch.from_numpy)
    return DataLoader(test_cross_sub_set, batch_size=None, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch, worker_init_fn=worker_init_fn)


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
        loader.worker_init_fn = worker_init_fn
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

    return loader


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
    print(cross_sub_test_ids)
    print(train_ids)

    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    train_loader = get_saved_train_loader(BATCH_SIZE, NUM_WORKERS)
    test_loader = get_saved_test_loader(BATCH_SIZE, NUM_WORKERS)
    test_cross_sub_loader = get_saved_test_cross_sub_loader(BATCH_SIZE, NUM_WORKERS)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Test_cross batches: {len(test_cross_sub_loader)}")

    loader = train_loader
    batches = len(loader)
    print(f"Batches in epoch: {batches}")

    for (i, item) in tqdm(enumerate(loader), total=batches):
        X, y = item
        print(f"batch: {i}/{batches},  X shape: {X.shape},  y shape: {y.shape}")
        # print(X.dtype)
        # print(y)
        # memory_usage_in_bytes = X.element_size() * X.nelement() + y.element_size() * y.nelement()
        # print(f"Memory Usage: {memory_usage_in_bytes} bytes")
