from pathlib import Path
import shutil
import numpy as np

from tqdm import tqdm
import yaml


# --- START OF CONSTANTS --- #
WINDOW_SAMPLES_SIZE = 512
N_SIGNALS = 2
DELETE_COMBINED_ARRAYS = False

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = config["paths"]["local"]["subset_1_directory"]
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")

ARRAYS_DIR = Path(PATH_TO_SUBSET1).joinpath("arrays")
EXPANDED_ARRAYS_DIR = Path(PATH_TO_SUBSET1).joinpath("arrays-expanded")

# --- END OF CONSTANTS --- #
# Get all ids in the directory with arrays. Each subdir is one subject
subset_ids = [int(f.name) for f in ARRAYS_DIR.iterdir() if f.is_dir()]


def get_subject_X_train(subject_id: int) -> np.array:
    X_path = ARRAYS_DIR.joinpath(str(subject_id).zfill(4)).joinpath("X_train.npy")
    # X = np.load(X_path).reshape(-1, WINDOW_SAMPLES_SIZE, N_SIGNALS).astype("float32")
    X = np.load(X_path)
    return X


def get_subject_y_train(subject_id: int) -> np.array:
    y_path = ARRAYS_DIR.joinpath(str(subject_id).zfill(4)).joinpath("y_train.npy")
    # y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")
    y = np.load(y_path)
    return y


def get_subject_X_test(subject_id: int) -> np.array:
    X_path = ARRAYS_DIR.joinpath(str(subject_id).zfill(4)).joinpath("X_test.npy")
    # X = np.load(X_path).reshape(-1, WINDOW_SAMPLES_SIZE, N_SIGNALS).astype("float32")
    X = np.load(X_path)
    return X


def get_subject_y_test(subject_id: int) -> np.array:
    y_path = ARRAYS_DIR.joinpath(str(subject_id).zfill(4)).joinpath("y_test.npy")
    # y = np.load(y_path).reshape(-1, WINDOW_SAMPLES_SIZE).astype("uint8")
    y = np.load(y_path)
    return y


for id in tqdm(subset_ids):
    X_train = get_subject_X_train(id)
    y_train = get_subject_y_train(id)
    continue
    n_train_windows = y_train.shape[0]
    for w in range(n_train_windows):
        X_window = X_train[w, :, :]
        y_window = y_train[w, :]
        print(f"X train - shape: {X_window.shape} type: {X_window.dtype}")
        print(f"y train - shape: {y_window.shape} type: {y_window.dtype}")

        # # Create directory for subject:
        # subject_train_dir = EXPANDED_ARRAYS_DIR.joinpath(str(id).zfill(4), "train")
        # subject_train_dir.mkdir(parents=True, exist_ok=True)
        #
        # X_window_path = subject_train_dir.joinpath(f"X_{w}.npy")
        # y_window_path = subject_train_dir.joinpath(f"y_{w}.npy")
        #
        # # Save the arrays
        # np.save(str(X_window_path), X_window)
        # np.save(str(y_window_path), y_window)

    X_test = get_subject_X_train(id)
    y_test = get_subject_y_train(id)
    n_test_windows = y_test.shape[0]
    for w in range(n_test_windows):
        X_window = X_test[w, :, :]
        y_window = y_test[w, :]
        print(f"X test - shape: {X_window.shape} type: {X_window.dtype}")
        print(f"y test - shape: {y_window.shape} type: {y_window.dtype}")

        # # Create directory for subject:
        # subject_test_dir = EXPANDED_ARRAYS_DIR.joinpath(str(id).zfill(4), "test")
        # subject_test_dir.mkdir(parents=True, exist_ok=True)
        #
        # X_window_path = subject_test_dir.joinpath(f"X_{w}.npy")
        # y_window_path = subject_test_dir.joinpath(f"y_{w}.npy")
        #
        # # Save the arrays
        # np.save(str(X_window_path), X_window)
        # np.save(str(y_window_path), y_window)

    if DELETE_COMBINED_ARRAYS:
        original_sub_arrays_dir = ARRAYS_DIR.joinpath(str(id).zfill(4))
        shutil.rmtree(original_sub_arrays_dir)
