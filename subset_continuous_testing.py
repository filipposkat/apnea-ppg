import math
from pathlib import Path
import numpy as np

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader, TensorDataset

# Local imports:
from common import Subject
from object_loader import get_subject_by_id, get_subjects_by_ids_generator
from subset_1_generator import get_best_ids
from trainer import load_checkpoint, get_last_batch, get_last_epoch

# --- START OF CONSTANTS --- #
SUBJECT_ID = 1212
NET_TYPE = "UResIncNet"
IDENTIFIER = "ks3-depth8-strided-0"
EPOCH = 35
CREATE_ARRAYS = False
SKIP_EXISTING_IDS = False
WINDOW_SEC_SIZE = 16
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
STEP = 512  # The step between each window
TEST_SIZE = 0.3
TEST_SEARCH_SAMPLE_STEP = 512
EXAMINED_TEST_SETS_SUBSAMPLE = 0.7  # Ratio of randomly selected test set candidates to all possible candidates
TARGET_TRAIN_TEST_SIMILARITY = 0.975  # Desired train-test similarity. 1=Identical distributions, 0=Completely different

SEED = 33

WINDOW_SAMPLES_SIZE = WINDOW_SEC_SIZE * SIGNALS_FREQUENCY

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])
    if CREATE_ARRAYS:
        PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{subset_id}_directory"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")


# --- END OF CONSTANTS --- #

def jensen_shannon_divergence(P: pd.Series, Q: pd.Series) -> float:
    """
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    :param P:
    :param Q:
    :return: The Jensen–Shannon divergence of the two distributions. JSD is a measure of dissimilarity. Bounds: 0<=JSD<=1
    """
    DPM = 0
    DMP = 0
    for x in range(5):
        Px = sum(P == x) / len(P)
        Qx = sum(Q == x) / len(Q)
        Mx = 0.5 * (Px + Qx)

        # Relative entropy from M to P:
        if Px == 0:
            DPM += 0
        else:
            DPM -= Px * math.log2(Mx / Px)

        # Relative entropy from M to Q:
        if Qx == 0:
            DMP += 0
        else:
            DMP -= Qx * math.log2(Mx / Qx)

    return 0.5 * DPM + 0.5 * DMP


def get_subject_continuous_test_data(subject: Subject, sufficiently_low_divergence=None) \
        -> tuple[list, list[pd.Series] | list[int]]:
    sub_df = subject.export_to_dataframe(signal_labels=["Pleth"], print_downsampling_details=False,
                                         max_frequency=SIGNALS_FREQUENCY)
    sub_df.drop(["time_secs"], axis=1, inplace=True)

    # 1. Do train test split preserving a whole sequence for test:
    # Find all possible sequences for test set:
    test_size = int(TEST_SIZE * sub_df.shape[0])  # number of rows/samples
    # Number of rolling test set sequences:
    num_of_candidates = (sub_df.shape[0] - test_size) // TEST_SEARCH_SAMPLE_STEP + 1  # a//b = math.floor(a/b)
    candidates = list(range(num_of_candidates))
    min_split_divergence = 99
    best_split = None

    candidates_subsample_size = int(EXAMINED_TEST_SETS_SUBSAMPLE * num_of_candidates)
    candidates_subsample = random.sample(candidates, k=candidates_subsample_size)

    sufficiently_low_divergence = 1.0 - TARGET_TRAIN_TEST_SIMILARITY
    for i in candidates_subsample:
        # Split into train and test:
        # Note: Continuity of train may break and dor this reason we want to keep the index of train intact
        # in order to know later the point where continuity breaks. So ignore_index=False
        train_df = pd.concat(
            [sub_df.iloc[:(i * TEST_SEARCH_SAMPLE_STEP)], sub_df.iloc[(i * TEST_SEARCH_SAMPLE_STEP + test_size):]],
            axis=0, ignore_index=False)
        test_df = sub_df.iloc[(i * TEST_SEARCH_SAMPLE_STEP):(i * TEST_SEARCH_SAMPLE_STEP + test_size)].reset_index(
            drop=True)

        # Find the JSD similarity of the train and test distributions:
        divergence = jensen_shannon_divergence(train_df["event_index"], test_df["event_index"])

        # We want to minimize the divergence because we want to maximize similarity
        if divergence < min_split_divergence:
            min_split_divergence = divergence
            best_split = (train_df, test_df)
            if divergence < sufficiently_low_divergence:
                break

    test_df = best_split[1]

    # Take equal-sized windows with a specified step:
    # 3. Calculate the number of windows:
    num_windows_test = len(test_df) // WINDOW_SAMPLES_SIZE

    # 4. Generate equal-sized windows:
    test_windows_dfs = [test_df.iloc[i * WINDOW_SAMPLES_SIZE:i * WINDOW_SAMPLES_SIZE + WINDOW_SAMPLES_SIZE]
                        for i in range(num_windows_test)]
    # Note that when using df.iloc[] or df[], the stop part is not included. However ,when using loc stop is included

    X_test = [window_df.loc[:, window_df.columns != "event_index"] for window_df in test_windows_dfs]
    y_test = [window_df["event_index"] for window_df in test_windows_dfs]
    return X_test, y_test


def save_arrays_combined(subject_arrs_path: Path, X_test, y_test):
    """
    Saves four arrays for one subject: X_train, y_train, X_test, y_test.
    X_train has shape (num of windows in train, WINDOW_SAMPLES_SIZE, numOfSignals)
    The order of signals is Flow and then Pleth.
    y_train has shape (num of windows in train, WINDOW_SAMPLES_SIZE, 1)
    X_test has shape (num of windows in test, WINDOW_SAMPLES_SIZE, numOfSignals+1)
    y_test has shape (num of windows in test, WINDOW_SAMPLES_SIZE, 1)

    :param subject_arrs_path: Path to save subject's arrays
    :param X_test: iterable with test window signals
    :param y_test: iterable with test window signals
    :return: Nothing
    """
    # Transform to numpy arrays:
    X_test_arr = np.array(X_test,
                          dtype="float32")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, numOfSignals+1)
    y_test_arr = np.array(y_test, dtype="uint8")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, 1)

    # Create directory for subject:
    subject_arrs_path.mkdir(parents=True, exist_ok=True)
    X_test_path = subject_arrs_path.joinpath("X_test")
    y_test_path = subject_arrs_path.joinpath("y_test")

    # Save the arrays
    np.save(str(X_test_path), X_test_arr)
    np.save(str(y_test_path), y_test_arr)


if __name__ == "__main__":
    if CREATE_ARRAYS:
        random.seed(SEED)  # Set the seed
        PATH_TO_SUBSET.mkdir(exist_ok=True)
        Path(PATH_TO_SUBSET).joinpath("cont-test-arrays").mkdir(exist_ok=True)
        best_ids = get_best_ids()
        print(best_ids)

        id, sub = get_subject_by_id(1212)
        X_test, y_test = get_subject_continuous_test_data(sub)
        subject_arrs_path = Path(PATH_TO_SUBSET).joinpath("cont-test-arrays", str(1212).zfill(4))
        save_arrays_combined(subject_arrs_path, X_test, y_test)

        for (id, sub) in get_subjects_by_ids_generator(best_ids, progress_bar=True):
            subject_arrs_path = Path(PATH_TO_SUBSET).joinpath("cont-test-arrays", str(id).zfill(4))

            if subject_arrs_path.exists() and SKIP_EXISTING_IDS:
                continue

            X_test, y_test = get_subject_continuous_test_data(sub)
            save_arrays_combined(subject_arrs_path, X_test, y_test)

    else:
        # id, sub = get_subject_by_id(SUBJECT_ID)
        # X_test, y_test = get_subject_continuous_test_data(sub)

        subject_arrs_path = Path(PATH_TO_SUBSET).joinpath("cont-test-arrays", str(SUBJECT_ID).zfill(4))
        X_path = subject_arrs_path.joinpath("X_test.npy")
        y_path = subject_arrs_path.joinpath("y_test.npy")

        X_test_arr = np.load(str(X_path)).astype("float32")
        y_test_arr = np.load(str(y_path)).astype("uint8")

        X_test_arr = np.swapaxes(X_test_arr, axis1=1, axis2=2)

        # Convert to tensors:
        X_test = torch.tensor(X_test_arr)
        y_test = torch.tensor(y_test_arr)

        # Testing
        dataset = TensorDataset(X_test, y_test)
        loader = DataLoader(dataset=dataset, batch_size=8192, shuffle=False)

        if torch.cuda.is_available():
            test_device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            test_device = torch.device("mps")
        else:
            test_device = torch.device("cpu")

        if EPOCH is None or EPOCH == "last":
            EPOCH = get_last_epoch(net_type=NET_TYPE, identifier=IDENTIFIER)
        batch = get_last_batch(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=EPOCH)
        net, _, _, _, _, _, _, _, _, _ = load_checkpoint(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=EPOCH,
                                                         batch=batch,
                                                         device=str(test_device))
        # Switch to eval mode:
        net.eval()
        net = net.to(test_device)
        saved_preds_for_stats = []
        saved_labels_for_stats = []
        with torch.no_grad():
            for (batch_i, data) in enumerate(loader):
                # get the inputs; data is a list of [inputs, labels]
                batch_inputs, batch_labels = data

                # Convert to accepted dtypes: float32, float64, int64 and maybe more but not sure
                batch_labels = batch_labels.type(torch.int64)

                batch_inputs = batch_inputs.to(test_device)
                batch_labels = batch_labels.to(test_device)

                # Predictions:
                batch_outputs = net(batch_inputs)
                # batch_output_probs = F.softmax(batch_outputs, dim=1)
                _, batch_predictions = torch.max(batch_outputs, dim=1, keepdim=False)

                saved_preds_for_stats.extend(batch_predictions.ravel().tolist())
                saved_labels_for_stats.extend(batch_labels.ravel().tolist())

        plt.figure()
        plt.scatter(list(range(len(saved_preds_for_stats))), saved_preds_for_stats, label="Predictions", s=0.1)
        plt.scatter(list(range(len(saved_labels_for_stats))), saved_labels_for_stats, label="True labels", s=0.1)
        plt.legend(loc='upper right')
        plt.ylabel("Class label")
        plt.xlabel("Sample i")
        plt.show()
        plt.figure()
        plt.plot(list(range(len(saved_preds_for_stats))), saved_preds_for_stats, label="Predictions")
        plt.plot(list(range(len(saved_labels_for_stats))), saved_labels_for_stats, label="True labels")
        plt.legend(loc='upper right')
        plt.ylabel("Class label")
        plt.xlabel("Sample i")
        plt.show()
        plt.close()
