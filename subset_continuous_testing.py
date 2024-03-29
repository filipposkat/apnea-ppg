import math
from pathlib import Path
import numpy as np
import scipy
from tqdm import tqdm

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Local imports:
from common import Subject
from object_loader import get_subject_by_id, get_subjects_by_ids_generator
from trainer import load_checkpoint, get_last_batch, get_last_epoch

# --- START OF CONSTANTS --- #
SUBJECT_ID = "all"  # 1212 lots obstructive, 5232 lots central
EPOCH = 32
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
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{subset_id}_training_directory"])
    if "subset_1_continuous_testing_directory" in config["paths"]["local"]:
        PATH_TO_SUBSET_CONT_TESTING = Path(config["paths"]["local"]["subset_1_continuous_testing_directory"])
    else:
        PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    MODELS_PATH = Path(config["paths"]["local"][f"subset_{subset_id}_saved_models_directory"])
    NET_TYPE = config["variables"]["models"]["net_type"]
    IDENTIFIER = config["variables"]["models"]["net_identifier"]
else:
    subset_id = 1
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_1_training_directory"])
    MODELS_PATH = Path(config["paths"]["local"][f"subset_1_saved_models_directory"])
    NET_TYPE = "UResIncNet"
    IDENTIFIER = "ks3-depth8-strided-0"


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


def get_subject_continuous_test_data(subject: Subject, sufficiently_low_divergence=None, split=True) \
        -> tuple[list, list[pd.Series] | list[int]]:
    sub_df = subject.export_to_dataframe(signal_labels=["Pleth"], print_downsampling_details=False,
                                         max_frequency=SIGNALS_FREQUENCY)
    sub_df.drop(["time_secs"], axis=1, inplace=True)

    if split:
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
        train_df = best_split[0]
        test_df = best_split[1]
    else:
        test_df = sub_df

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

    path = PATH_TO_SUBSET.joinpath("ids.npy")
    if path.is_file():
        best_ids_arr = np.load(str(path))  # array to save the best subject ids
        best_ids = best_ids_arr.tolist()  # equivalent list
    else:
        print(f"Subset-{subset_id} has no ids generated yet")
        exit(1)

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
    if CREATE_ARRAYS:
        random.seed(SEED)  # Set the seed
        PATH_TO_SUBSET_CONT_TESTING.mkdir(exist_ok=True)
        PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-arrays").mkdir(exist_ok=True)
        print(best_ids)

        for (id, sub) in get_subjects_by_ids_generator(best_ids, progress_bar=True):
            subject_arrs_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-arrays", str(id).zfill(4))

            if subject_arrs_path.exists() and SKIP_EXISTING_IDS:
                continue

            if id in train_ids:
                split = True
            else:
                split = False
            X_test, y_test = get_subject_continuous_test_data(sub, split=split)
            save_arrays_combined(subject_arrs_path, X_test, y_test)

    else:
        # id, sub = get_subject_by_id(SUBJECT_ID)
        # X_test, y_test = get_subject_continuous_test_data(sub)

        if SUBJECT_ID == "all":
            sub_ids = best_ids
        elif isinstance(SUBJECT_ID, int):
            sub_ids = [SUBJECT_ID]
        elif isinstance(SUBJECT_ID, list):
            sub_ids = SUBJECT_ID
        else:
            exit(1)

        for sub_id in tqdm(sub_ids):
            subject_arrs_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-arrays", str(sub_id).zfill(4))
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
            saved_probs_for_stats = []
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
                    batch_output_probs = F.softmax(batch_outputs, dim=1)
                    _, batch_predictions = torch.max(batch_outputs, dim=1, keepdim=False)

                    saved_preds_for_stats.extend(batch_predictions.ravel().tolist())
                    saved_probs_for_stats.extend(batch_output_probs.swapaxes(1, 2).reshape(-1, 5).tolist())
                    saved_labels_for_stats.extend(batch_labels.ravel().tolist())

            results_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-results", str(NET_TYPE), str(IDENTIFIER), f"epoch-{EPOCH}")
            if sub_id in train_ids:
                results_path = results_path.joinpath("validation-subjects")
            else:
                results_path = results_path.joinpath("cross-test-subjects")

            results_path.mkdir(parents=True, exist_ok=True)
            matlab_file = results_path.joinpath(f"cont_test_signal_{sub_id}.mat")
            matlab_dict = {"prediction_probabilities": np.array(saved_probs_for_stats),
                           "predictions": np.array(saved_preds_for_stats),
                           "labels": np.array(saved_labels_for_stats)}
            scipy.io.savemat(matlab_file, matlab_dict)

            # plt.figure()
            # plt.scatter(list(range(len(saved_preds_for_stats))), saved_preds_for_stats, label="Predictions", s=0.1)
            # plt.scatter(list(range(len(saved_labels_for_stats))), saved_labels_for_stats, label="True labels", s=0.1)
            # plt.title(f"Continuous Prediction for f{sub_id}")
            # plt.legend(loc='upper right')
            # plt.ylabel("Class label")
            # plt.xlabel("Sample i")
            # plt.show()
            # plt.figure()
            # plt.plot(list(range(len(saved_preds_for_stats))), saved_preds_for_stats, label="Predictions")
            # plt.plot(list(range(len(saved_labels_for_stats))), saved_labels_for_stats, label="True labels")
            # plt.title(f"Continuous Prediction for f{sub_id}")
            # plt.legend(loc='upper right')
            # plt.ylabel("Class label")
            # plt.xlabel("Sample i")
            # plt.show()
            # plt.close()
