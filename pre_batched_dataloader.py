from pathlib import Path
from typing import Callable, Any
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset

# Local imports:
from data_loaders_iterable import IterDataset, worker_init_fn, \
    get_saved_train_loader, get_saved_test_loader, get_saved_test_cross_sub_loader


BATCH_SIZE = 128
SEED = 33
NUM_WORKERS = 4
SKIP_EXISTING = True
SKIP_TRAIN = False
SKIP_TEST = False
SKIP_CROSS_TEST = False

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1

ARRAYS_DIR = PATH_TO_SUBSET1.joinpath("arrays")

# Paths for saving batch tensors:
PRE_BATCHED_TENSORS_PATH = PATH_TO_SUBSET1_TRAINING.joinpath("pre-batched-tensors", f"bs{BATCH_SIZE}")


def create_pre_batched_tensors(batch_size=BATCH_SIZE):
    train_loader = get_saved_train_loader(batch_size, num_workers=NUM_WORKERS)
    test_loader = get_saved_test_loader(batch_size, num_workers=NUM_WORKERS)
    cross_test_loader = get_saved_test_cross_sub_loader(batch_size, num_workers=NUM_WORKERS)

    train_tensors_path = PRE_BATCHED_TENSORS_PATH.joinpath("train")
    test_tensors_path = PRE_BATCHED_TENSORS_PATH.joinpath("test")
    cross_test_tensors_path = PRE_BATCHED_TENSORS_PATH.joinpath("test-cross-subject")
    train_tensors_path.mkdir(parents=True, exist_ok=True)
    test_tensors_path.mkdir(parents=True, exist_ok=True)
    cross_test_tensors_path.mkdir(parents=True, exist_ok=True)

    if not SKIP_TRAIN:
        print("Saving batches from train loader:")
        batches = len(train_loader)
        for (i, item) in tqdm(enumerate(train_loader), total=batches):
            batch_path = train_tensors_path.joinpath(f"batch-{i}")

            # if batch has been saved already then skip it:
            if SKIP_EXISTING and batch_path.exists() and train_tensors_path.joinpath(f"batch-{i + 1}").exists():
                # The last saved batch (=> batch-(i+1) does not exist) should not be skipped because it may be corrupted.
                continue

            batch_path.mkdir(exist_ok=True)
            X_path = batch_path.joinpath("X.pt")
            y_path = batch_path.joinpath("y.pt")
            X, y = item
            torch.save(X, X_path)
            torch.save(y, y_path)

    if not SKIP_TEST:
        batches = len(test_loader)
        for (i, item) in tqdm(enumerate(test_loader), total=batches):
            batch_path = test_tensors_path.joinpath(f"batch-{i}")
            batch_path.mkdir(exist_ok=True)
            X_path = batch_path.joinpath("X.pt")
            y_path = batch_path.joinpath("y.pt")
            X, y = item
            torch.save(X, X_path)
            torch.save(y, y_path)

    if not SKIP_CROSS_TEST:
        batches = len(cross_test_loader)
        for (i, item) in tqdm(enumerate(cross_test_loader), total=batches):
            batch_path = cross_test_tensors_path.joinpath(f"batch-{i}")
            batch_path.mkdir(exist_ok=True)
            X_path = batch_path.joinpath("X.pt")
            y_path = batch_path.joinpath("y.pt")
            X, y = item
            torch.save(X, X_path)
            torch.save(y, y_path)


class PreBatchedDataset(Dataset):

    def __init__(self, batches_dir: Path):
        self.directory = batches_dir

    def __len__(self):
        count = 0
        for path in self.directory.iterdir():
            if path.is_dir():
                count += 1
        return count

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """
        :param idx: Batch index
        :return: (signal, labels) where signal = Pleth signal of shape (
        """

        batch_dir = self.directory / f"batch-{idx}"
        X_path = batch_dir / "X.pt"
        y_path = batch_dir / "y.pt"
        X = torch.load(X_path)
        y = torch.load(y_path)
        return X, y


def adaptive_collate_fn(batches: list[tuple[torch.tensor, torch.tensor]]) -> tuple[torch.tensor, torch.tensor]:
    """
    :param batches: List of tuples with two elements: X batched tensor and y batched tensor.
     X batched tensors have  shape: (batch_size, 1, WINDOW_SIZE)
     y batched tensors have shape: (batch_size, WINDOW_SIZE)
    :return: Two tensors (X,y), the concatenation of all batches
    """
    input_batches = [batch[0] for batch in batches]
    label_batches = [batch[1] for batch in batches]

    combined_input_batches = torch.cat(input_batches, dim=0)
    combined_label_batches = torch.cat(label_batches, dim=0)
    return combined_input_batches, combined_label_batches


def get_pre_batched_train_loader(batch_size: int = BATCH_SIZE, n_workers=NUM_WORKERS) -> DataLoader:
    assert batch_size % BATCH_SIZE == 0

    dir_path = PRE_BATCHED_TENSORS_PATH / "train"
    train_set = PreBatchedDataset(dir_path)

    batch_multiplier = batch_size // BATCH_SIZE
    rng = torch.Generator()
    rng.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=batch_multiplier, collate_fn=adaptive_collate_fn, shuffle=True,
                              generator=rng, pin_memory=True, num_workers=n_workers)
    return train_loader


def get_pre_batched_test_loader(batch_size: int = BATCH_SIZE, n_workers=NUM_WORKERS) -> DataLoader:
    assert batch_size % BATCH_SIZE == 0

    dir_path = PRE_BATCHED_TENSORS_PATH / "test"
    train_set = PreBatchedDataset(dir_path)

    batch_multiplier = batch_size // BATCH_SIZE
    rng = torch.Generator()
    rng.manual_seed(SEED)
    test_loader = DataLoader(train_set, batch_size=batch_multiplier, collate_fn=adaptive_collate_fn, shuffle=True,
                             generator=rng, pin_memory=True, num_workers=n_workers)
    return test_loader


def get_pre_batched_test_cross_sub_loader(batch_size: int = BATCH_SIZE, n_workers=NUM_WORKERS) -> DataLoader:
    assert batch_size % BATCH_SIZE == 0

    dir_path = PRE_BATCHED_TENSORS_PATH / "test-cross-subject"
    train_set = PreBatchedDataset(dir_path)

    batch_multiplier = batch_size // BATCH_SIZE

    rng = torch.Generator()
    rng.manual_seed(SEED)
    test_cross_sub_loader = DataLoader(train_set, batch_size=batch_multiplier, collate_fn=adaptive_collate_fn,
                                       shuffle=True, generator=rng, pin_memory=True, num_workers=n_workers)
    return test_cross_sub_loader


if __name__ == "__main__":
    create_pre_batched_tensors(batch_size=BATCH_SIZE)
