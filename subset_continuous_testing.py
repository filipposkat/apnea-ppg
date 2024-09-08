import json
import math
from pathlib import Path
from typing import Literal
import pickle
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
from data_loaders_mapped import get_subject_train_test_split

# --- START OF CONSTANTS --- #
TESTING_SUBSET = 0
SUBJECT_ID = "all"  # 1212 lots obstructive, 5232 lots central
EPOCH = 10
CREATE_ARRAYS = False
GET_CONTINUOUS_PREDICTIONS = False
SKIP_EXISTING_IDS = True
PER_WINDOW_EVALUATION = True

# CREATE ARRAYS PARAMS:
WINDOW_SEC_SIZE = 16
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
STEP = 512  # The step between each window
TEST_SIZE = 0.3
TEST_SEARCH_SAMPLE_STEP = 512
EXAMINED_TEST_SETS_SUBSAMPLE = 0.7  # Ratio of randomly selected test set candidates to all possible candidates
TARGET_TRAIN_TEST_SIMILARITY = 0.975  # Desired train-test similarity. 1=Identical distributions, 0=Completely different

# Per window TESTING PARAMS:
AGGREGATION_WINDOW_SIZE_SECS = 1
NORMALIZE: Literal["true", "pred", "all", "none"] = "none"
DERSIRED_MERGED_CLASSES = 2
SEED = 33

WINDOW_SAMPLES_SIZE = WINDOW_SEC_SIZE * SIGNALS_FREQUENCY

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])
    if "convert_spo2desat_to_normal" in config["variables"]["dataset"]:
        CONVERT_SPO2DESAT_TO_NORMAL = config["variables"]["dataset"]["convert_spo2desat_to_normal"]
    else:
        CONVERT_SPO2DESAT_TO_NORMAL = False

    if "n_input_channels" in config["variables"]["dataset"]:
        N_INPUT_CHANNELS = config["variables"]["dataset"]["n_input_channels"]
    else:
        N_INPUT_CHANNELS = 1

    if CREATE_ARRAYS:
        PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{TESTING_SUBSET}_directory"])
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{subset_id}_training_directory"])
    if f"subset_{TESTING_SUBSET}_continuous_testing_directory" in config["paths"]["local"]:
        PATH_TO_SUBSET_CONT_TESTING = Path(
            config["paths"]["local"][f"subset_{TESTING_SUBSET}_continuous_testing_directory"])
    else:
        PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    MODELS_PATH = Path(config["paths"]["local"][f"subset_{subset_id}_saved_models_directory"])
    NET_TYPE = config["variables"]["models"]["net_type"]
    IDENTIFIER = config["variables"]["models"]["net_identifier"]
else:
    subset_id = 1
    CONVERT_SPO2DESAT_TO_NORMAL = False
    N_INPUT_CHANNELS = 1
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_1_training_directory"])
    MODELS_PATH = Path(config["paths"]["local"][f"subset_1_saved_models_directory"])
    NET_TYPE = "UResIncNet"
    IDENTIFIER = "ks3-depth8-strided-0"


# --- END OF CONSTANTS --- #
def get_window_label(window_labels: np.ndarray):
    if np.sum(window_labels) == 0:
        return 0, 1.0
    else:
        # 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation

        unique_events, event_counts = np.unique(window_labels, return_counts=True)
        prominent_event_index = np.argmax(event_counts)
        prominent_event = unique_events[prominent_event_index]
        confidence = event_counts[prominent_event_index] / len(window_labels)
        return int(prominent_event), float(confidence)


def get_window_label2(window_labels: np.ndarray, frequency=SIGNALS_FREQUENCY):
    """
    :param window_labels:
    :param frequency:
    :return: If at least 10s of apneic event is inside the window, then it is labeled as such. Otherwise,
    it is labeled based on the highest event count.
    """
    if np.sum(window_labels) == 0:
        return 0, 1.0
    else:
        # 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation

        unique_events, event_counts = np.unique(window_labels, return_counts=True)
        prominent_event_index = np.argmax(event_counts)
        prominent_event = unique_events[prominent_event_index]
        confidence = event_counts[prominent_event_index] / len(window_labels)

        if len(window_labels) >= 10 * frequency and not (1 <= prominent_event <= 3):
            apneic_labels = window_labels[(1 <= window_labels) & (window_labels <= 3)]
            if len(apneic_labels) > 10 * frequency:
                unique_apneic_events, apneic_event_counts = np.unique(apneic_labels, return_counts=True)
                prominent_apneic_event_index = np.argmax(apneic_event_counts)
                prominent_apneic_event = unique_events[prominent_event_index]
                prominent_apneic_event_count = apneic_event_counts[prominent_apneic_event_index]
                if prominent_apneic_event_count >= 10 * frequency:
                    prominent_event = prominent_apneic_event
                    confidence = prominent_apneic_event_count / len(window_labels)

        return int(prominent_event), float(confidence)


def get_window_label_from_probas(window_probas: np.ndarray):
    """
    :param window_probas: Array of shape (window length, n classes)
    :return:
    """
    agg_probas = np.mean(window_probas, axis=0)
    prominent_event_index = np.argmax(agg_probas)
    confidence = agg_probas[prominent_event_index]
    return int(prominent_event_index), float(confidence)


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


def get_subject_continuous_test_data(subject: Subject, split=True, train_test_split_index: int = None) \
        -> tuple[list, list[pd.Series] | list[int]]:
    sub_df = subject.export_to_dataframe(signal_labels=["Pleth"], print_downsampling_details=False,
                                         frequency=SIGNALS_FREQUENCY)
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
        if train_test_split_index is None:
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
            i = train_test_split_index
            test_df = sub_df.iloc[(i * TEST_SEARCH_SAMPLE_STEP):(i * TEST_SEARCH_SAMPLE_STEP + test_size)].reset_index(
                drop=True)
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
        subset_ids_arr = np.load(str(path))  # array to save the best subject ids
        subset_ids = subset_ids_arr.tolist()  # equivalent list
    else:
        print(f"Subset-{subset_id} has no ids generated yet")
        exit(1)

    if subset_id == 1:
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
    else:
        train_ids, _ = get_subject_train_test_split()
    print(train_ids)

    if CREATE_ARRAYS:
        random.seed(SEED)  # Set the seed
        PATH_TO_SUBSET_CONT_TESTING.mkdir(exist_ok=True)
        PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-arrays").mkdir(exist_ok=True)
        print(subset_ids)

        # These dicts should be provided to get accurate validation splits.
        # The files needs to be in the TESTING_SUBSET dir
        rng_seed_pth = PATH_TO_SUBSET / "sub_seed_dict.plk"
        saved_sub_seed_dict = {}
        saved_split_index_dict = {}
        split_index_pth = PATH_TO_SUBSET / "train_test_split_index_dict.plk"
        if rng_seed_pth.is_file():
            with open(str(rng_seed_pth), mode="rb") as file:
                saved_sub_seed_dict = pickle.load(file)
        if split_index_pth.is_file():
            # Not first time generating the subset
            with open(str(split_index_pth), mode="rb") as file:
                saved_split_index_dict = pickle.load(file)

        for (id, sub) in get_subjects_by_ids_generator(subset_ids, progress_bar=True):
            subject_arrs_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-arrays", str(id).zfill(4))
            subject_arrs_path.mkdir(exist_ok=True)

            # Save metadata
            metadata = sub.metadata
            metadata_df = pd.Series(metadata)
            metadata_df.to_csv(subject_arrs_path.joinpath("sub_metadata.csv"))

            if subject_arrs_path.exists() and SKIP_EXISTING_IDS:
                continue

            if id in train_ids:
                split = True
            else:
                split = False

            if id in saved_sub_seed_dict:
                state = saved_sub_seed_dict[id]
                random.setstate(state)

            if split and id in saved_split_index_dict:
                train_test_split_i = saved_split_index_dict[id]
            else:
                train_test_split_i = None

            X_test, y_test = get_subject_continuous_test_data(sub, split=split,
                                                              train_test_split_index=train_test_split_i)
            save_arrays_combined(subject_arrs_path, X_test, y_test)

    elif GET_CONTINUOUS_PREDICTIONS:
        # id, sub = get_subject_by_id(SUBJECT_ID)
        # X_test, y_test = get_subject_continuous_test_data(sub)

        if SUBJECT_ID == "all":
            sub_ids = subset_ids
        elif isinstance(SUBJECT_ID, int):
            sub_ids = [SUBJECT_ID]
        elif isinstance(SUBJECT_ID, list):
            sub_ids = SUBJECT_ID
        else:
            exit(1)

        for sub_id in tqdm(sub_ids):
            results_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-results", str(NET_TYPE), str(IDENTIFIER),
                                                                f"epoch-{EPOCH}")
            if sub_id in train_ids:
                results_path = results_path.joinpath("validation-subjects")
            else:
                results_path = results_path.joinpath("cross-test-subjects")

            results_path.mkdir(parents=True, exist_ok=True)
            matlab_file = results_path.joinpath(f"cont_test_signal_{sub_id}.mat")
            if SKIP_EXISTING_IDS and matlab_file.exists():
                continue

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

            sample_batch_input, sample_batch_labels = next(iter(loader))
            window_size = sample_batch_input.shape[2]
            N_CLASSES = int(torch.max(sample_batch_labels)) + 1
            if CONVERT_SPO2DESAT_TO_NORMAL:
                assert N_CLASSES == 5
                N_CLASSES = 4

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

                    if CONVERT_SPO2DESAT_TO_NORMAL:
                        batch_labels[batch_labels == 4] = 0

                    n_channels_found = batch_inputs.shape[1]
                    if n_channels_found != N_INPUT_CHANNELS:
                        diff = n_channels_found - N_INPUT_CHANNELS
                        assert diff > 0
                        # Excess channels have been detected, exclude the first (typically SpO2 or Flow)
                        batch_inputs = batch_inputs[:, diff:, :]

                    # Predictions:
                    batch_outputs = net(batch_inputs)
                    batch_output_probs = F.softmax(batch_outputs, dim=1)
                    _, batch_predictions = torch.max(batch_outputs, dim=1, keepdim=False)

                    saved_preds_for_stats.extend(batch_predictions.ravel().tolist())
                    saved_probs_for_stats.extend(batch_output_probs.swapaxes(1, 2).reshape(-1, N_CLASSES).tolist())
                    saved_labels_for_stats.extend(batch_labels.ravel().tolist())

            matlab_dict = {"prediction_probabilities": np.array(saved_probs_for_stats),
                           "predictions": np.array(saved_preds_for_stats),
                           "labels": np.array(saved_labels_for_stats, dtype="uint8"),
                           "trained_subject": sub_id in train_ids}
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
    else:
        from sklearn.metrics import confusion_matrix
        from torchmetrics import ROC, AUROC
        from adjusted_test_stats import get_stats_from_cm, get_metrics_from_cm, classification_performance, \
            merged_classes_assesment, merge_sum_columns

        if SUBJECT_ID == "all":
            sub_ids = subset_ids
        elif isinstance(SUBJECT_ID, int):
            sub_ids = [SUBJECT_ID]
        elif isinstance(SUBJECT_ID, list):
            sub_ids = SUBJECT_ID
        else:
            exit(1)

        results_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-results", str(NET_TYPE), str(IDENTIFIER),
                                                            f"epoch-{EPOCH}")
        agg_path = results_path / f"aggregation_win_{AGGREGATION_WINDOW_SIZE_SECS}s"
        agg_path.mkdir(exist_ok=True)

        # prepare to count predictions for each class
        all_classes = ("normal", "central_apnea", "obstructive_apnea", "hypopnea", "spO2_desat")
        classes = None

        if agg_path.joinpath("agg_cm2.json").exists():
            with open(agg_path / "agg_cm2.json", 'r') as file:
                agg_cm2 = np.array(json.load(file))
            with open(agg_path / "aggregate_roc_info.json", 'r') as file:
                roc_info_by_class = json.load(file)

            n_class = agg_cm2.shape[0]
            classes = all_classes[0:n_class]
            classification_performance(cm=agg_cm2, plot_confusion=True, target_labels=classes, normalize=NORMALIZE)
            merged_metrics = merged_classes_assesment(agg_cm2, desired_classes=DERSIRED_MERGED_CLASSES,
                                                      normalize=NORMALIZE)
            # Save merged metrics
            with open(agg_path / f"agg_merged{DERSIRED_MERGED_CLASSES}_metrics.json", 'w') as file:
                json.dump(merged_metrics, file)

            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            axs = axs.ravel()

            # Save the ROC plot:
            plot_path = agg_path.joinpath(f"aggregate_roc.png")
            for c, class_name in enumerate(roc_info_by_class.keys()):
                average_fpr = roc_info_by_class[class_name]["average_fpr"]
                average_tpr = roc_info_by_class[class_name]["average_tpr"]
                average_auc = roc_info_by_class[class_name]["average_auc"]

                ax = axs[c]
                ax.plot(average_fpr, average_tpr)
                ax.set_title(f"Average ROC for class: {class_name} with average AUC: {average_auc:.2f}")
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")

            fig.savefig(str(plot_path))
            plt.show()
            plt.close(fig)
        else:
            agg_cm1: np.ndarray
            agg_cm2: np.ndarray
            agg_metrics_1 = {}
            agg_metrics_2 = {}

            thresholds = torch.tensor(np.linspace(start=0, stop=1, num=100))
            tprs_by_class = {}
            fprs_by_class = {}
            aucs_by_class = {}

            for sub_id in tqdm(sub_ids):
                if sub_id in train_ids:
                    sub_path = results_path.joinpath("validation-subjects")
                else:
                    sub_path = results_path.joinpath("cross-test-subjects")

                assert sub_path.exists()

                matlab_file = sub_path.joinpath(f"cont_test_signal_{sub_id}.mat")

                matlab_dict = scipy.io.loadmat(str(matlab_file))
                prediction_probas: np.ndarray = matlab_dict["prediction_probabilities"]
                predictions: np.ndarray = matlab_dict["predictions"]
                labels: np.ndarray = matlab_dict["labels"]

                predictions = np.squeeze(predictions)
                labels = np.squeeze(labels)

                n_class = prediction_probas.shape[1]
                if classes is None:
                    classes = [all_classes[c] for c in range(n_class)]
                    agg_cm1 = np.zeros((n_class, n_class))
                    agg_cm2 = np.zeros((n_class, n_class))
                    tprs_by_class = {c: [] for c in classes}
                    fprs_by_class = {c: [] for c in classes}
                    aucs_by_class = {c: [] for c in classes}

                length = len(labels)

                aggregation_window_sample_size = AGGREGATION_WINDOW_SIZE_SECS * SIGNALS_FREQUENCY
                n = length // aggregation_window_sample_size
                remainder = length % aggregation_window_sample_size

                seg_predictions = np.split(predictions[remainder:], indices_or_sections=n)
                seg_prediction_probas = np.split(prediction_probas[remainder:, :], indices_or_sections=n, axis=0)
                seg_labels = np.split(labels[remainder:], indices_or_sections=n)

                per_window_preds1 = np.array([get_window_label(win)[0] for win in seg_predictions])
                per_window_preds2 = np.array([get_window_label_from_probas(win)[0] for win in seg_prediction_probas])
                per_window_probas = np.array([np.mean(win, axis=0) for win in seg_prediction_probas])
                per_window_labels = np.array([get_window_label(win)[0] for win in seg_labels])

                # RoC curve for this subject:
                roc = ROC(task="multiclass", thresholds=thresholds, num_classes=n_class)
                auroc = AUROC(task="multiclass", thresholds=thresholds, num_classes=n_class, average="none")
                fprs, tprs, _ = roc(torch.tensor(per_window_probas), torch.tensor(per_window_labels, dtype=torch.int64))
                aucs = auroc(torch.tensor(per_window_probas), torch.tensor(per_window_labels, dtype=torch.int64))

                for c in range(n_class):
                    class_name = classes[c]
                    fprs_by_class[class_name].append(fprs[c, :])
                    tprs_by_class[class_name].append(tprs[c, :])
                    aucs_by_class[class_name].append(aucs[c])

                # Confusion matrix for this subject
                sub_cm1 = confusion_matrix(y_true=per_window_labels, y_pred=per_window_preds1,
                                           labels=np.arange(n_class))
                sub_cm2 = confusion_matrix(y_true=per_window_labels, y_pred=per_window_preds2,
                                           labels=np.arange(n_class))
                agg_cm1 += sub_cm1
                agg_cm2 += sub_cm2

                # Metrics from CM:
                metrics1 = get_metrics_from_cm(cm=sub_cm1, classes=classes)
                metrics2 = get_metrics_from_cm(cm=sub_cm2, classes=classes)

                for (k, v) in metrics1.items():
                    if isinstance(v, dict):
                        if k not in agg_metrics_1:
                            agg_metrics_1[k] = {c: 0 for c in classes}
                            agg_metrics_2[k] = {c: 0 for c in classes}

                        for (ks, vs) in v.items():
                            v1 = 0 if vs == "nan" else vs
                            v2 = 0 if metrics2[k][ks] == "nan" else metrics2[k][ks]

                            agg_metrics_1[k][ks] += v1
                            agg_metrics_2[k][ks] += v2
                    else:
                        if k not in agg_metrics_1:
                            agg_metrics_1[k] = 0
                            agg_metrics_2[k] = 0

                        agg_metrics_1[k] += v
                        agg_metrics_2[k] += metrics2[k]

                # classification_performance(cm=sub_cm1, plot_confusion=True, target_labels=classes)
                # classification_performance(cm=sub_cm2, plot_confusion=True, target_labels=classes)

            # Compute threshold average ROC for each class, across subjected:
            roc_info_by_class = {}
            average_auc_by_class = {}
            for class_name in classes:
                fprs = torch.stack(fprs_by_class[class_name], dim=0)
                average_fpr = torch.mean(fprs, dim=0, keepdim=False)
                tprs = torch.stack(tprs_by_class[class_name], dim=0)
                average_tpr = torch.mean(tprs, dim=0, keepdim=False)
                aucs = torch.tensor(aucs_by_class[class_name])
                average_auc = torch.mean(aucs)

                average_auc_by_class[class_name] = average_auc.item()

                roc_info_by_class[class_name] = {
                    "thresholds": thresholds.tolist(),
                    "average_fpr": average_fpr.tolist(),
                    "average_tpr": average_tpr.tolist(),
                    "average_auc": average_auc.item()
                }

            # Save ROC info:
            with open(agg_path / "aggregate_roc_info.json", 'w') as file:
                json.dump(roc_info_by_class, file)

            # Save the ROC plot:
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            axs = axs.ravel()

            plot_path = agg_path.joinpath(f"aggregate_roc.png")
            for c, class_name in enumerate(roc_info_by_class.keys()):
                average_fpr = roc_info_by_class[class_name]["average_fpr"]
                average_tpr = roc_info_by_class[class_name]["average_tpr"]
                average_auc = roc_info_by_class[class_name]["average_auc"]

                ax = axs[c]
                ax.plot(average_fpr, average_tpr)
                ax.set_title(f"Average ROC for class: {class_name} with average AUC: {average_auc:.2f}")
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
            fig.savefig(str(plot_path))
            plt.close(fig)

            # Average metrics across subjects
            for (k, v) in agg_metrics_1.items():
                if isinstance(v, dict):
                    for ks in v.keys():
                        agg_metrics_1[k][ks] /= len(sub_ids)
                        agg_metrics_2[k][ks] /= len(sub_ids)
                else:
                    agg_metrics_1[k] /= len(sub_ids)
                    agg_metrics_2[k] /= len(sub_ids)

            classification_performance(agg_cm2, plot_confusion=True, target_labels=classes, normalize=NORMALIZE)

            merged_metrics = merged_classes_assesment(agg_cm2, desired_classes=DERSIRED_MERGED_CLASSES,
                                                      normalize=NORMALIZE)

            print(agg_metrics_1)
            print(agg_metrics_2)

            with open(agg_path / "agg_cm1.json", 'w') as file:
                json.dump(agg_cm1.tolist(), file)

            with open(agg_path / "agg_cm2.json", 'w') as file:
                json.dump(agg_cm2.tolist(), file)

            with open(agg_path / "agg_metrics_1.json", 'w') as file:
                json.dump(agg_metrics_1, file)

            with open(agg_path / "agg_metrics_2.json", 'w') as file:
                json.dump(agg_metrics_2, file)

            # Save merged metrics
            with open(agg_path / f"agg_merged{DERSIRED_MERGED_CLASSES}_metrics.json", 'w') as file:
                json.dump(merged_metrics, file)
