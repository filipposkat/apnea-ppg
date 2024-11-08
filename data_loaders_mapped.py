from itertools import cycle
from pathlib import Path
import yaml
import json
import numpy as np
import random
import pickle
from sortedcontainers import SortedList

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

GENERATE_TRAIN_TEST_SPLIT = False
CROSS_SUBJECT_TEST_PROPORTION = 0.25
BATCH_WINDOW_SAMPLING_RATIO = 0.1
BATCH_SIZE = 256
BATCH_SIZE_TEST = 1024  # 8192
SEED = 33
NUM_WORKERS = 2  # better to use power of two, otherwise each worker will have different number of subject ids
PREFETCH_FACTOR = 1
INCLUDE_TRAIN_IN_CROSS_SUB_TESTING = False
SAVE_BATCH_INDICES_TRAIN = False
SAVE_BATCH_INDICES_TEST = True
SAVE_BATCH_INDICES_CROSS_TEST = True

CROSS_SUBJECT_TEST_SIZE = 100
with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)
if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{subset_id}_directory"])
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{subset_id}_training_directory"])
    if "n_input_channels" in config["variables"]["dataset"]:
        N_DESIRED_INPUT_SIGNALS = config["variables"]["dataset"]["n_input_channels"]
    else:
        N_DESIRED_INPUT_SIGNALS = 1
else:
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_TRAINING = PATH_TO_SUBSET
    N_DESIRED_INPUT_SIGNALS = 1

ARRAYS_DIR = PATH_TO_SUBSET.joinpath("arrays")

# Paths for saving dataloaders:
dataloaders_path = PATH_TO_SUBSET_TRAINING.joinpath("dataloaders-mapped")
dataloaders_path.mkdir(parents=True, exist_ok=True)


def get_subject_train_test_split(save_ids=False):
    path = PATH_TO_SUBSET / "ids.npy"
    if path.is_file():
        ids_arr = np.load(str(path))  # array to save the best subject ids
        ids: list = ids_arr.tolist()  # equivalent list
    else:
        print(f"Subset-{subset_id} has no ids generated yet")
        exit(1)

    if GENERATE_TRAIN_TEST_SPLIT or len(ids) != 400:
        cross_sub_test_size = int(len(ids) * CROSS_SUBJECT_TEST_PROPORTION)
        rng = random.Random(33)
        cross_sub_test_ids = rng.sample(ids, cross_sub_test_size)
        train_ids = [id for id in ids if id not in cross_sub_test_ids]

        if save_ids:
            split_dict = {"train_ids": train_ids, "cross_testing_ids": cross_sub_test_ids}
            dict_path = PATH_TO_SUBSET_TRAINING / "train_cross_test_split_ids.json"
            with open(str(dict_path), "w") as file:
                json.dump(split_dict, file)
    else:
        cross_sub_test_ids = [5002, 1453, 5396, 2030, 2394, 4047, 5582, 4478, 4437, 1604, 6726, 5311, 4229, 2780, 5957,
                              6697, 4057, 3823, 2421, 5801, 5451, 679, 2636, 3556, 2688, 4322, 4174, 572, 5261, 5847,
                              3671,
                              2408, 2771, 4671, 5907, 2147, 979, 620, 6215, 2434, 1863, 651, 3043, 1016, 5608, 6538,
                              2126,
                              4270, 2374, 6075, 107, 3013, 4341, 5695, 2651, 6193, 3332, 3314, 1589, 935, 386, 3042,
                              5393,
                              4794, 6037, 648, 1271, 811, 1010, 2750, 33, 626, 3469, 6756, 2961, 1756, 1650, 3294, 3913,
                              5182, 4014, 3025, 5148, 4508, 3876, 2685, 4088, 675, 125, 6485, 3239, 5231, 3037, 5714,
                              5986,
                              155, 4515, 6424, 2747, 1356]
        train_ids = [27, 64, 133, 140, 183, 194, 196, 220, 303, 332, 346, 381, 405, 407, 435, 468, 490, 505, 527, 561,
                     571,
                     589, 628, 643, 658, 712, 713, 715, 718, 719, 725, 728, 743, 744, 796, 823, 860, 863, 892, 912, 917,
                     931, 934, 937, 939, 951, 1013, 1017, 1019, 1087, 1089, 1128, 1133, 1161, 1212, 1224, 1236, 1263,
                     1266,
                     1278, 1281, 1291, 1301, 1328, 1342, 1376, 1464, 1478, 1497, 1501, 1502, 1552, 1562, 1573, 1623,
                     1626,
                     1656, 1693, 1733, 1738, 1790, 1797, 1809, 1833, 1838, 1874, 1879, 1906, 1913, 1914, 1924, 1983,
                     2003,
                     2024, 2039, 2105, 2106, 2118, 2204, 2208, 2216, 2227, 2239, 2246, 2251, 2264, 2276, 2291, 2292,
                     2317,
                     2345, 2375, 2397, 2451, 2452, 2467, 2468, 2523, 2539, 2572, 2614, 2665, 2701, 2735, 2781, 2798,
                     2800,
                     2802, 2819, 2834, 2848, 2877, 2879, 2881, 2897, 2915, 2934, 2995, 3012, 3024, 3028, 3106, 3149,
                     3156,
                     3204, 3223, 3236, 3275, 3280, 3293, 3324, 3337, 3347, 3352, 3419, 3439, 3452, 3468, 3555, 3564,
                     3575,
                     3591, 3603, 3604, 3652, 3690, 3702, 3711, 3734, 3743, 3770, 3781, 3803, 3833, 3852, 3854, 3867,
                     3902,
                     3933, 3934, 3967, 3974, 3980, 3987, 3992, 4029, 4038, 4085, 4099, 4123, 4128, 4157, 4163, 4205,
                     4228,
                     4250, 4252, 4254, 4256, 4295, 4296, 4330, 4332, 4428, 4462, 4496, 4497, 4511, 4541, 4544, 4554,
                     4592,
                     4624, 4661, 4734, 4820, 4826, 4878, 4912, 4948, 5029, 5053, 5063, 5075, 5096, 5101, 5118, 5137,
                     5155,
                     5162, 5163, 5179, 5203, 5214, 5232, 5276, 5283, 5308, 5339, 5357, 5358, 5365, 5387, 5395, 5433,
                     5457,
                     5472, 5480, 5491, 5503, 5565, 5580, 5662, 5686, 5697, 5703, 5753, 5788, 5798, 5845, 5897, 5909,
                     5954,
                     5982, 6009, 6022, 6047, 6050, 6052, 6074, 6077, 6117, 6174, 6180, 6244, 6261, 6274, 6279, 6280,
                     6291,
                     6316, 6318, 6322, 6351, 6366, 6390, 6417, 6422, 6492, 6528, 6549, 6616, 6682, 6695, 6704, 6755,
                     6781,
                     6804, 6807, 6811]

    return train_ids, cross_sub_test_ids


class MappedDataset(Dataset):

    def __init__(self,
                 subject_ids: list[int],
                 dataset_split: str = "train",
                 desired_target: str = "y",
                 arrays_dir: Path = ARRAYS_DIR,
                 transform=None,
                 target_transform=None,
                 initialization_progress=True) -> None:

        self.subject_ids = subject_ids
        self.target = desired_target
        self.arrays_dir = arrays_dir
        self.dataset_split = dataset_split

        if self.dataset_split == "train":
            self.load_arrays = self.train_array_loader
        elif self.dataset_split == "test":
            self.load_arrays = self.test_array_loader
        else:
            self.load_arrays = self.cross_test_array_loader

        self.transform = transform
        self.target_transform = target_transform

        # Determine n_signals / channels (if pleth only =1 if pleth and flow then =2):
        sample_id: int = self.subject_ids[0]
        X, _ = self.load_arrays(sample_id)
        n_signals = X.shape[1]

        # assert n_signals == 1  # Flow has been deleted already
        # assert X.shape[2] == WINDOW_SAMPLES_SIZE

        # Inputs are indexed with two numbers (id, window_index),
        self.id_size_dict = {}
        self.total_windows = 0

        subject_ids = tqdm(self.subject_ids) if initialization_progress else self.subject_ids
        for id in subject_ids:
            X, y = self.load_arrays(id)
            detected_n_signals = X.shape[1]
            if detected_n_signals != N_DESIRED_INPUT_SIGNALS:
                print(f"Subject: {id} has incorrect number of signals. Expected: {N_DESIRED_INPUT_SIGNALS}, Got: {detected_n_signals} ")
            # detected_window_size_= X.shape[2]

            n_windows = y.shape[0]
            self.id_size_dict[id] = n_windows
            self.total_windows += n_windows

    def train_array_loader(self, sub_id: int) -> tuple[np.array, np.array]:
        X_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("X_train.npy")
        y_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("y_train.npy")

        X = np.load(str(X_path)).astype("float32")  # shape: (n_windows, window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=1, axis2=2)  # shape: (n_windows, n_signals, window_size), Flow comes first

        if "flow" == self.target:
            assert X.shape[1] >= 2
            y = X[:, 0, :]
            X = np.delete(X, 0, axis=1)
        else:
            y = np.load(str(y_path)).astype("uint8")
            if y.ndim != 1:
                y = y.reshape(X.shape[0], X.shape[2])

        if N_DESIRED_INPUT_SIGNALS < X.shape[1]:
            assert X.shape[1] == N_DESIRED_INPUT_SIGNALS + 1
            # Drop the flow signal:
            X = np.delete(X, 0, axis=1)
        return X, y

    def test_array_loader(self, sub_id: int) -> tuple[np.array, np.array]:
        X_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("X_test.npy")
        y_path = self.arrays_dir.joinpath(str(sub_id).zfill(4)).joinpath("y_test.npy")

        X = np.load(str(X_path)).astype("float32")  # shape: (n_windows, window_size, n_signals), Flow comes first
        X = np.swapaxes(X, axis1=1, axis2=2)  # shape: (n_windows, n_signals, window_size), Flow comes first

        if "flow" == self.target:
            assert X.shape[1] >= 2
            y = X[:, 0, :]
            X = np.delete(X, 0, axis=1)
        else:
            y = np.load(str(y_path)).astype("uint8")
            if y.ndim != 1:
                y = y.reshape(X.shape[0], X.shape[2])

        if N_DESIRED_INPUT_SIGNALS < X.shape[1]:
            assert X.shape[1] == N_DESIRED_INPUT_SIGNALS + 1
            # Drop the flow signal:
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
        if y.ndim == 1:
            labels = y[windows_indices]
        else:
            labels = y[windows_indices, :]
        return signals, labels

    def __len__(self):
        # Find out the total windows:
        self.total_windows = 0
        for id in self.subject_ids:
            self.total_windows += self.id_size_dict[id]

        return self.total_windows

    def __getitem__(self, batch_windows_by_sub: dict[int: list[int]]) -> tuple[np.array, np.array]:
        """
        :param batch_windows_by_sub: Dict: {subject_id: [window_index1, window_index2, ...]}
        :return: (signal, labels) where shape of signal: (batch_size, 1, window_size)
        and of labels: (batch_size, window_size)
        """

        X_batch = []
        y_batch = []
        for sub_id in batch_windows_by_sub.keys():
            window_indices = batch_windows_by_sub[sub_id]
            signals, labels = self.get_specific_windows(sub_id, window_indices)
            if signals.shape[1] != N_DESIRED_INPUT_SIGNALS:
                print(f"Subject: {sub_id} has incorrect number of signals. Expected: {N_DESIRED_INPUT_SIGNALS}, Got: {signals.shape[1]} ")
            X_batch.append(signals)
            y_batch.append(labels)

        # batch_2d_indices = sorted(batch_2d_indices)
        # windows_indices = []
        # for i in range(len(batch_2d_indices)):
        #     index_2d = batch_2d_indices[i]
        #     sub_id = index_2d[0]
        #     windows_indices.append(index_2d[1])
        #
        #     # Check if the next id is different:
        #     if i == len(batch_2d_indices) - 1 or sub_id != batch_2d_indices[i + 1][0]:
        #         signals, labels = self.get_specific_windows(sub_id, windows_indices)
        #         X_batch.append(signals)
        #         y_batch.append(labels)
        #         windows_indices = []

        X_batch = np.concatenate(X_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)
        return X_batch, y_batch


class BatchSampler(Sampler[list[int]]):
    def __init__(self, subject_ids: list[int], batch_size: int, id_size_dict: dict[int: int],
                 shuffle=False, seed=None, pbar=False) -> None:
        """
        :param subject_ids:
        :param batch_size:
        :param id_size_dict:
        :param shuffle: Whether to shuffle the subjects between epochs
        :param seed: The seed to use for shuffling and sampling
        """

        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.first_batch_index = 0

        self.shuffle = shuffle
        if seed is not None:
            self.rng = random.Random(seed)
            self.seed = seed
        else:
            self.rng = random.Random()
            self.seed = None

        self.id_size_dict = id_size_dict
        self.pbar = pbar

    def __len__(self) -> int:
        # Find out the total windows:
        self.total_windows = 0
        for id in self.subject_ids:
            self.total_windows += self.id_size_dict[id]

        # Number of batches:
        return (self.total_windows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = 0
        # # by default tuples are sorted with preference given to the first elements (subject in this case):
        # used_2d_indices = SortedList()

        used_windows_by_sub: dict[int: SortedList] = {}
        for sub_id in self.subject_ids:
            used_windows_by_sub[sub_id] = SortedList()

        # Shuffle ids in-place:
        if self.shuffle:
            self.rng.shuffle(self.subject_ids)
            self.rng.getstate()

        # Cyclic iterator of our ids:
        pool = cycle(self.subject_ids)
        total_batches = len(self)

        pbar = None
        if self.pbar:
            pbar = tqdm(total=total_batches)

        while batch < total_batches - 1:
            ids_tmp = []
            sub_id1 = next(pool)
            sub_id2 = next(pool)
            sub_id3 = next(pool)

            # Get the number of windows in the third subject, this will be needed to calculate the last index
            n_windows1 = self.id_size_dict[sub_id1]
            n_windows2 = self.id_size_dict[sub_id2]
            n_windows3 = self.id_size_dict[sub_id3]

            # indices1 = [(sub_id1, i) for i in range(n_windows1) if (sub_id1, i) not in used_2d_indices]
            # indices2 = [(sub_id2, i) for i in range(n_windows2) if (sub_id2, i) not in used_2d_indices]
            # indices3 = [(sub_id3, i) for i in range(n_windows3) if (sub_id3, i) not in used_2d_indices]
            if len(used_windows_by_sub[sub_id1]) == n_windows1:
                indices1 = []
            else:
                indices1 = [(sub_id1, i) for i in range(n_windows1) if i not in used_windows_by_sub[sub_id1]]
                ids_tmp.append(sub_id1)

            if len(used_windows_by_sub[sub_id2]) == n_windows2:
                indices2 = []
            else:
                indices2 = [(sub_id2, i) for i in range(n_windows2) if i not in used_windows_by_sub[sub_id2]]
                ids_tmp.append(sub_id2)

            if len(used_windows_by_sub[sub_id3]) == n_windows3:
                indices3 = []
            else:
                indices3 = [(sub_id3, i) for i in range(n_windows3) if i not in used_windows_by_sub[sub_id3]]
                ids_tmp.append(sub_id3)

            combined_indices = [*indices1, *indices2, *indices3]

            # Check if the windows available for sampling are enough for a batch:
            extra_indices = []
            while len(combined_indices) + len(extra_indices) < self.batch_size:
                # If three subjects do not have enough windows then use more:
                next_sub_id = next(pool)

                # Get the number of windows in the third subject, this will be needed to calculate the last index
                n_windows = self.id_size_dict[next_sub_id]

                if used_windows_by_sub[next_sub_id] == n_windows:
                    continue
                # indices_to_add = [(next_sub_id, i) for i in range(n_windows) if (next_sub_id, i) not in used_2d_indices]
                indices_to_add = [(next_sub_id, i) for i in range(n_windows) if
                                  i not in used_windows_by_sub[next_sub_id]]

                extra_indices.extend(indices_to_add)
                ids_tmp.append(next_sub_id)

            # Sample windows:
            if len(combined_indices) >= self.batch_size:
                batch_2d_indices = self.rng.sample(combined_indices, self.batch_size)
            else:
                extra_indices = self.rng.sample(extra_indices, self.batch_size - len(combined_indices))
                batch_2d_indices = [*combined_indices, *extra_indices]

            if len(batch_2d_indices) != self.batch_size:
                print("Indices less than batch size")

            # Transform 2d_index to dict and save used batches:
            batch_windows_by_sub: dict[int: list] = {}
            for sub_id in ids_tmp:
                batch_windows_by_sub[sub_id] = []

            for index_2d in batch_2d_indices:
                sub_id = index_2d[0]
                window_id = index_2d[1]
                batch_windows_by_sub[sub_id].append(window_id)
                used_windows_by_sub[sub_id].add(window_id)

            # used_2d_indices.update(batch_2d_indices)

            # Check if this batch should be yielded or skipped
            if batch >= self.first_batch_index:
                yield batch_windows_by_sub
                # yield batch_2d_indices

            batch += 1
            if self.pbar:
                pbar.update(1)

        # The last batch may contain less than the batch_size:
        # unused_2d_indices = []
        unused_windows_by_sub: dict[int: list] = {}
        for sub_id in self.subject_ids:
            n_windows = self.id_size_dict[sub_id]
            for window_index in range(n_windows):
                if window_index not in used_windows_by_sub[sub_id]:
                    if sub_id not in unused_windows_by_sub.keys():
                        unused_windows_by_sub[sub_id] = [window_index]
                    else:
                        unused_windows_by_sub[sub_id].append(window_index)
                # if (sub_id, window_index) not in used_2d_indices:
                #     unused_2d_indices.append((sub_id, window_index))

        yield unused_windows_by_sub

        if self.pbar:
            pbar.update(1)
            pbar.close()
        # assert len(unused_2d_indices) <= self.batch_size
        # yield unused_2d_indices


class BatchFromSavedBatchIndices(Sampler[list[int]]):
    def __init__(self, batch_indices_path: Path, shuffle=False, seed=None) -> None:
        """
        :param shuffle: Whether to shuffle the subjects between epochs
        :param seed: The seed to use for shuffling and sampling
        """

        self.first_batch_index = 0

        self.shuffle = shuffle
        if seed is not None:
            self.rng = random.Random(seed)
            self.seed = seed
        else:
            self.rng = random.Random()
            self.seed = None

        self.batch_indices_path = batch_indices_path

    def __len__(self) -> int:
        batches = 0
        for _ in self.batch_indices_path.iterdir():
            batches += 1

        # Number of batches:
        return batches

    def __iter__(self):
        batch_ids = []

        for batch_indices_file in self.batch_indices_path.iterdir():
            batch_id = int(batch_indices_file.name.split('-')[1])
            batch_ids.append(batch_id)

        batch_ids = sorted(batch_ids)

        # Shuffle ids in-place:
        if self.shuffle:
            self.rng.shuffle(batch_ids)
        for i, batch_id in enumerate(batch_ids):
            if i >= self.first_batch_index:
                batch_windows_by_sub: dict
                batch_indices_file = self.batch_indices_path / f"batch-{batch_id}"
                with open(batch_indices_file, mode="rb") as f:
                    batch_windows_by_sub = pickle.load(f)
                    f.close()
                yield batch_windows_by_sub


def get_new_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR, shuffle=True) \
        -> DataLoader:
    train_ids, _ = get_subject_train_test_split(save_ids=True)
    dataset = MappedDataset(subject_ids=train_ids,
                            dataset_split="train",
                            transform=torch.from_numpy,
                            target_transform=torch.from_numpy)

    sampler = BatchSampler(subject_ids=train_ids, batch_size=batch_size, id_size_dict=dataset.id_size_dict,
                           shuffle=shuffle, seed=SEED)

    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    return DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch)


def get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR,
                           arrays_dir: Path = None, use_existing_batch_indices=False, shuffle=True) -> DataLoader:
    if NUM_WORKERS == 0:
        pre_fetch = None

    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Mapped_Train_Loader.pickle")
    if not object_file.exists():
        loader = get_new_train_loader(batch_size=batch_size, num_workers=num_workers, pre_fetch=pre_fetch,
                                      shuffle=shuffle)

        # Save train loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

    if object_file.is_file():
        with open(object_file, "rb") as f:
            loader = pickle.load(f)

        loader.sampler.shuffle = shuffle
        loader.num_workers = num_workers
        loader.prefetch_factor = pre_fetch

        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

        if use_existing_batch_indices:
            batch_indices_path = dataloaders_path / f"bs{batch_size}" / "batch-indices" / "train"
            if batch_indices_path.is_dir():
                dataset = loader.dataset

                sampler = BatchFromSavedBatchIndices(batch_indices_path=batch_indices_path,
                                                     shuffle=loader.sampler.shuffle,
                                                     seed=loader.sampler.seed)
                loader = DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True,
                                    num_workers=num_workers, prefetch_factor=pre_fetch)

    return loader


def save_batch_indices_train(batch_size=BATCH_SIZE):
    loader = get_saved_train_loader(batch_size=batch_size, num_workers=1, pre_fetch=1, use_existing_batch_indices=False)
    sampler = loader.sampler
    batch_indices_path = dataloaders_path / f"bs{batch_size}" / "batch-indices" / "train"
    batch_indices_path.mkdir(exist_ok=True, parents=True)
    sampler.pbar = True

    batch = 0
    for batch_windows_by_sub in sampler:
        thisbatch_path = batch_indices_path / f"batch-{batch}"
        with open(thisbatch_path, mode='wb') as f:
            pickle.dump(batch_windows_by_sub, f)
        batch += 1


def get_new_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR, shuffle=False) \
        -> DataLoader:
    train_ids, _ = get_subject_train_test_split()
    dataset = MappedDataset(subject_ids=train_ids,
                            dataset_split="test",
                            transform=torch.from_numpy,
                            target_transform=torch.from_numpy)

    sampler = BatchSampler(subject_ids=train_ids, batch_size=batch_size, id_size_dict=dataset.id_size_dict,
                           shuffle=shuffle, seed=SEED)

    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    return DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch)


def get_saved_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR,
                          arrays_dir: Path = None, use_existing_batch_indices=False, shuffle=False) -> DataLoader:
    if NUM_WORKERS == 0:
        pre_fetch = None

    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Mapped_Test_Loader.pickle")
    if not object_file.exists():
        loader = get_new_test_loader(batch_size=batch_size, num_workers=num_workers, pre_fetch=pre_fetch,
                                     shuffle=shuffle)

        # Save test loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

    if object_file.is_file():
        with open(object_file, "rb") as f:
            loader = pickle.load(f)

        loader.sampler.shuffle = shuffle
        loader.num_workers = num_workers
        loader.prefetch_factor = pre_fetch
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

        if use_existing_batch_indices:
            dataset = loader.dataset
            batch_indices_path = dataloaders_path / f"bs{batch_size}" / "batch-indices" / "test"
            if batch_indices_path.is_dir():
                sampler = BatchFromSavedBatchIndices(batch_indices_path=batch_indices_path,
                                                     shuffle=loader.sampler.shuffle,
                                                     seed=loader.sampler.seed)
                loader = DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True,
                                    num_workers=num_workers, prefetch_factor=pre_fetch)

    return loader


def save_batch_indices_test(batch_size=BATCH_SIZE):
    loader = get_saved_test_loader(batch_size=batch_size, num_workers=1, pre_fetch=1, use_existing_batch_indices=False)
    sampler = loader.sampler
    batch_indices_path = dataloaders_path / f"bs{batch_size}" / "batch-indices" / "test"
    batch_indices_path.mkdir(exist_ok=True, parents=True)
    sampler.pbar = True

    batch = 0
    for batch_windows_by_sub in sampler:
        thisbatch_path = batch_indices_path / f"batch-{batch}"
        with open(thisbatch_path, mode='wb') as f:
            pickle.dump(batch_windows_by_sub, f)
        batch += 1


def get_new_test_cross_sub_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS,
                                  pre_fetch=PREFETCH_FACTOR, shuffle=False) -> DataLoader:
    _, cross_sub_test_ids = get_subject_train_test_split()
    dataset = MappedDataset(subject_ids=cross_sub_test_ids,
                            dataset_split="cross_test",
                            transform=torch.from_numpy,
                            target_transform=torch.from_numpy)

    sampler = BatchSampler(subject_ids=cross_sub_test_ids, batch_size=batch_size, id_size_dict=dataset.id_size_dict,
                           shuffle=shuffle, seed=SEED)

    # It is important to set batch_size=None which disables automatic batching,
    # because dataset returns them batched already:
    return DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True,
                      num_workers=num_workers, prefetch_factor=pre_fetch)


def get_saved_test_cross_sub_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR,
                                    arrays_dir: Path = None, use_existing_batch_indices=False, shuffle=False) \
        -> DataLoader:
    if NUM_WORKERS == 0:
        pre_fetch = None

    dataloaders_path.joinpath(f"bs{batch_size}").mkdir(parents=True, exist_ok=True)
    object_file = dataloaders_path.joinpath(f"bs{batch_size}",
                                            f"PlethToLabel_Mapped_TestCrossSub_Loader.pickle")
    if not object_file.exists():
        loader = get_new_test_cross_sub_loader(batch_size=batch_size, num_workers=num_workers, pre_fetch=pre_fetch)

        # Save cross test loader for future use
        with open(object_file, "wb") as file:
            pickle.dump(loader, file)

    if object_file.is_file():
        with open(object_file, "rb") as f:
            loader = pickle.load(f)

        loader.sampler.shuffle = shuffle
        loader.num_workers = num_workers
        loader.prefetch_factor = pre_fetch
        if arrays_dir:
            loader.dataset.arrays_dir = arrays_dir

        if use_existing_batch_indices:
            batch_indices_path = dataloaders_path / f"bs{batch_size}" / "batch-indices" / "cross_test"
            if batch_indices_path.is_dir():
                dataset = loader.dataset
                sampler = BatchFromSavedBatchIndices(batch_indices_path=batch_indices_path,
                                                     shuffle=loader.sampler.shuffle,
                                                     seed=loader.sampler.seed)
                loader = DataLoader(dataset, batch_size=None, sampler=sampler, pin_memory=True,
                                    num_workers=num_workers, prefetch_factor=pre_fetch)

    return loader


def save_batch_indices_test_cross_sub(batch_size=BATCH_SIZE):
    loader = get_saved_test_cross_sub_loader(batch_size=batch_size, num_workers=1, pre_fetch=1,
                                             use_existing_batch_indices=False)
    sampler = loader.sampler
    batch_indices_path = dataloaders_path / f"bs{batch_size}" / "batch-indices" / "cross_test"
    batch_indices_path.mkdir(exist_ok=True, parents=True)
    sampler.pbar = True

    batch = 0
    for batch_windows_by_sub in sampler:
        thisbatch_path = batch_indices_path / f"batch-{batch}"
        with open(thisbatch_path, mode='wb') as f:
            pickle.dump(batch_windows_by_sub, f)
        batch += 1


if __name__ == "__main__":
    train_ids, cross_sub_test_ids = get_subject_train_test_split()
    print(train_ids)
    print(cross_sub_test_ids)

    train_loader = get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                          pre_fetch=PREFETCH_FACTOR, use_existing_batch_indices=False)
    test_loader = get_saved_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS,
                                        pre_fetch=PREFETCH_FACTOR)
    test_cross_sub_loader = get_saved_test_cross_sub_loader(batch_size=BATCH_SIZE_TEST,
                                                            num_workers=NUM_WORKERS, pre_fetch=PREFETCH_FACTOR)

    print(f"Train batches: {len(train_loader)}  bs: {BATCH_SIZE}")
    print(f"Test batches: {len(test_loader)}  bs: {BATCH_SIZE_TEST}")
    print(f"Test_cross batches: {len(test_cross_sub_loader)}  bs: {BATCH_SIZE_TEST}")
    # train_loader.sampler.first_batch_index = 51303

    if SAVE_BATCH_INDICES_TRAIN:
        save_batch_indices_train(batch_size=BATCH_SIZE)
    if SAVE_BATCH_INDICES_TEST:
        save_batch_indices_test(batch_size=BATCH_SIZE_TEST)
    if SAVE_BATCH_INDICES_CROSS_TEST:
        save_batch_indices_test_cross_sub(batch_size=BATCH_SIZE_TEST)

    for (i, item) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X, y = item
        print(f"batch: {i},  X shape: {X.shape},  y shape: {y.shape}")
        print(f"X dtype: {X.dtype}  y dtype: {y.dtype}")
    #     print(X)
    #     print(y)
    # print(X.dtype)
    # print(y)
    # memory_usage_in_bytes = X.element_size() * X.nelement() + y.element_size() * y.nelement()
    # print(f"Memory Usage: {memory_usage_in_bytes} bytes")
