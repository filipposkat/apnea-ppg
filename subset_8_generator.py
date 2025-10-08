# Novel subset that contains all subjects but with only Train and Cross-Test.
# It includes equal representation of all OSA categories.
# Uses more advanced preprocessing
# Uses extra signals derived from PPG
# Has much less overlapping to limit the otherwise huge size
# Test windows have no overlap

import math
import time
from itertools import filterfalse, count
from pathlib import Path
from typing import Tuple, Any, List
import numpy as np
from collections import Counter
from tqdm import tqdm
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import json
from sklearn.model_selection import train_test_split

# Local imports:
from common import Subject, detect_desaturations_profusion
from object_loader import all_subjects_generator, get_subjects_by_ids_generator, get_subject_by_id, get_all_ids

# --- START OF CONSTANTS --- #
# SIGNALS = ["Pleth", "Slow_Pleth", "Pleth_Envelope", "Pleth_KTE", "SpO2"]
SIGNALS = ["Pleth", "Slow_Pleth", "Pleth_Envelope", "SpO2"]
CLEAN_SPO2 = True
CONVERT_SPO2_TO_DST_LABELS = True
EXCLUDE_LOW_SQI_SUBJECTS_FROM_TRAIN = True
EXCLUDE_LOW_TST_SUBJECTS_FROM_TRAIN = True
SUBSET = 8
SUBSET_SIZE = "all"  # The number of subjects that will remain after screening down the whole dataset
CREATE_ARRAYS = True
SKIP_EXISTING_IDS = False  # NOT RECOMMENDED, does not yield the same train-test splits in subjects!
WINDOW_SEC_SIZE = 60
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
ANTI_ALIASING = True
TRIM = True
SCALE_SIGNALS = True  # Normalizes signal with percentile method
STEP_SECS = int(0.1 * WINDOW_SEC_SIZE)  # The step between each window
CONTINUOUS_LABEL = True
NO_EVENTS_TO_EVENTS_RATIO = 1
INCLUDE_SPO2DESAT_IN_NOEVENT = True
MIN_WINDOWS = 1000  # Minimum value of subject's windows to remain after window dropping
EXCLUDE_10s_AFTER_EVENT = True
DROP_EVENT_WINDOWS_IF_NEEDED = False
COUNT_LABELS = True
SAVE_ARRAYS_EXPANDED = False
SEED = 33

WINDOW_SAMPLES_SIZE = WINDOW_SEC_SIZE * SIGNALS_FREQUENCY
STEP = STEP_SECS * SIGNALS_FREQUENCY
with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{SUBSET}_directory"])
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{SUBSET}_training_directory"])
    PATH_TO_METADATA = Path(config["paths"]["local"]["subject_metadata_file"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", f"subset-{SUBSET}")
    PATH_TO_SUBSET_TRAINING = Path(__file__).parent.joinpath("data", f"subset-{SUBSET}")
    PATH_TO_METADATA = Path(__file__).parent.joinpath("data", "mesa-original", "datasets",
                                                      "mesa-sleep-dataset-0.7.0.csv")


# --- END OF CONSTANTS --- #
def ahi_to_category(ahi: float) -> int:
    if ahi < 5.0:
        # No
        return 0
    elif ahi < 15.0:
        # Mild
        return 1
    elif ahi < 30.0:
        # Moderate
        return 2
    else:
        # Severe
        return 3


def get_ids():
    """
    Splits subjects into 40% train, 10% validation and 50% test.
    However, train may eventually contain less than 40% due to rejection criteria based on TST and SQI
    :return:
    """
    path = Path(PATH_TO_SUBSET).joinpath("ids.npy")
    dict_path = PATH_TO_SUBSET_TRAINING / "train_cross_test_split_ids.json"
    if path.is_file() and dict_path.is_file():
        ids_arr = np.load(str(path))  # array to save the best subject ids
        ids = ids_arr.tolist()  # equivalent list
        with open(str(dict_path), "r") as file:
            split_dict = json.load(file)
    else:
        df: pd.DataFrame = pd.read_csv(PATH_TO_METADATA, sep=',')
        df.dropna(
            subset=["slpprdp5", "qupleth5", "ahi_a0h3a", "ahi_o0h3a", "ahi_c0h3a", "gender1", "race1c", "sleepage5c"],
            inplace=True, ignore_index=True)
        df.set_index(keys="mesaid", drop=False, inplace=True)
        df.index.names = [None]
        all_sub_ids = get_all_ids()
        df = df.loc[all_sub_ids, :]
        print(df.shape)

        df["mesaid"] = df["mesaid"].astype("int64")
        df["slpprdp5"] = df["slpprdp5"].astype("int64")
        df["qupleth5"] = df["qupleth5"].astype("int64")
        df["ahi_a0h3a"] = df["ahi_a0h3a"].astype(float)
        df["ahi_o0h3a"] = df["ahi_o0h3a"].astype(float)
        df["ahi_c0h3a"] = df["ahi_c0h3a"].astype(float)
        sqi_l3 = df[df["qupleth5"] < 3.0]["mesaid"].tolist()
        tst_l3 = df[df["slpprdp5"] < 3.0 * 60]["mesaid"].tolist()
        train_blacklist = [*sqi_l3, *tst_l3]
        df["ahi_cat"] = df["ahi_a0h3a"].map(ahi_to_category)
        df["ahi_o_cat"] = df["ahi_o0h3a"].map(ahi_to_category)
        df["ahi_c_cat"] = df["ahi_c0h3a"].map(ahi_to_category)

        df = df.loc[:, ["mesaid", "gender1", "race1c", "ahi_cat", "ahi_o_cat", "ahi_c_cat"]]

        # TrainVal-Test Split: We want to stratify based on Race, Sex, AHI
        df['race_sex_ahi'] = df['race1c'].astype(str) + "_" + df['gender1'].astype(str) + "_" + df['ahi_cat'].astype(
            str)
        train_df, test_df = train_test_split(df,
                                             test_size=0.5,
                                             stratify=df['race_sex_ahi'],
                                             shuffle=True,
                                             random_state=SEED)
        test_ids = test_df["mesaid"].tolist()

        # Train-Val Split: We want to stratify based on Race, Sex, AHI
        train_df, validation_df = train_test_split(train_df,
                                                   test_size=0.2,
                                                   stratify=train_df['race_sex_ahi'],
                                                   shuffle=True,
                                                   random_state=SEED)
        train_ids = train_df["mesaid"].tolist()
        val_ids = validation_df["mesaid"].tolist()

        # Reject subjects based on bad quality criteria:
        train_ids = [id for id in train_ids if id not in train_blacklist]

        ids = [*train_ids, *val_ids, *test_ids]
        ids_arr = np.array(ids)  # Equivalent array
        path = Path(PATH_TO_SUBSET).joinpath("ids")
        np.save(str(path), ids_arr)
        path = PATH_TO_SUBSET.joinpath("ids.csv")
        ids_arr.tofile(str(path), sep=',')

        split_dict = {"train_ids": train_ids,
                      "validation_ids": val_ids,
                      "cross_testing_ids": test_ids}
        dict_path = PATH_TO_SUBSET_TRAINING / "train_cross_test_split_ids.json"
        with open(str(dict_path), "w") as file:
            json.dump(split_dict, file)
    return ids.copy(), split_dict


# best_ids = get_best_ids()
# print(f"Final subset size: {len(best_ids)}\n")
# print(best_ids)


def assign_window_label(window_labels: pd.Series) -> int:
    """
    :param window_labels: A pandas Series with the event labels for each sample in the window
    :return: One label representing the whole window, specifically the label with most occurrences.
    """
    if not isinstance(window_labels, pd.Series):
        window_labels = pd.Series(window_labels)

    if window_labels.sum() == 0:
        return 0
    else:
        # 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation
        event_counts = window_labels.value_counts()  # Series containing counts of unique values.
        prominent_event_index = event_counts.argmax()
        prominent_event = event_counts.index[prominent_event_index]
        # print(event_counts)
        return int(prominent_event)


def relative_entropy(P: pd.Series, Q: pd.Series) -> float:
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    Dkl = 0  # Kullback–Leibler divergence or relative entropy
    for x in range(5):
        Px = sum(P == x) / len(P)
        Qx = sum(Q == x) / len(Q)
        if Px == 0:
            Dkl += 0
        else:
            Dkl -= Px * math.log2(Qx / Px)

    return Dkl


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


def get_subject_windows(subject: Subject, train=True) \
        -> tuple[Any, Any]:
    sub_df = subject.export_to_dataframe(signal_labels=SIGNALS, print_downsampling_details=False,
                                         anti_aliasing=True, frequency=SIGNALS_FREQUENCY, trim_signals=TRIM,
                                         clean_spo2=CLEAN_SPO2,
                                         detrend_ppg=True)
    sub_df.drop(["time_secs"], axis=1, inplace=True)

    # Ensure correct order of signals:
    sub_df = sub_df.loc[:, [*SIGNALS, "event_index"]]

    if CONVERT_SPO2_TO_DST_LABELS:
        _, dst_lbls = detect_desaturations_profusion(sub_df["SpO2"], sampling_rate=SIGNALS_FREQUENCY,
                                                     return_label_series=True)
        sub_df.drop(["SpO2"], axis=1, inplace=True)
        sub_df["DST_Labels"] = dst_lbls
        sub_df["DST_Labels"] = sub_df["DST_Labels"].astype("uint8")

    samples_to_exclude = []
    if train and EXCLUDE_10s_AFTER_EVENT:
        # 1. Drop no-event train samples within 10s (= 320 samples) after an event:

        # for every sample
        for i in sub_df.index:
            # if it has event:
            if sub_df.loc[i, "event_index"] != 0:
                # Check the next 10s for no event samples:
                blacklist_tmp = []
                for j in range(i + 1, i + 321):
                    # Check if continuity breaks at j (due to train test split):
                    if j not in sub_df.index:
                        # if j is not in index it means that train is not continuous everywhere because of the train
                        # test split. Consequently, train is continuous on two parts (one part before and one part after
                        # the test). This means that j-1 was the last point of the first part.
                        break
                    if sub_df.loc[j, "event_index"] == 0:
                        # Add no event sample to blacklist:
                        blacklist_tmp.append(j)
                    else:
                        # Event sample found within 10s of the event sample that we examine,
                        # thus there is to examine for more no event samples, since i will reach that event.
                        # Also, the blacklisting of these samples will not be enforced in order to keep this event.
                        blacklist_tmp.clear()
                        break
                samples_to_exclude.extend(blacklist_tmp)

    # Take equal-sized windows with a specified step:
    # 2. Calculate the number of windows:
    if train:
        num_windows = (len(sub_df) - WINDOW_SAMPLES_SIZE) // STEP + 1  # a//b = math.floor(a/b)
        # Note that due to floor division the last WINDOW_SAMPLES_SIZE-1 samples might be dropped

        # 3. Generate equal-sized windows:
        windows_dfs = [sub_df.iloc[i * STEP:i * STEP + WINDOW_SAMPLES_SIZE]
                       for i in range(num_windows)]
        # Note that when using df.iloc[] or df[], the stop part is not included. However,when using loc stop is included
    else:
        num_windows = len(sub_df) // WINDOW_SAMPLES_SIZE  # a//b = math.floor(a/b)
        windows_dfs = [sub_df.iloc[i * WINDOW_SAMPLES_SIZE:i * WINDOW_SAMPLES_SIZE + WINDOW_SAMPLES_SIZE]
                       for i in range(num_windows)]
        # Note that when using df.iloc[] or df[], the stop part is not included. However,when using loc stop is included

    X = [window_df.loc[:, window_df.columns != "event_index"] for window_df in windows_dfs]

    # 4. Drop no-event windows to achieve desired ratio of no-events to events:
    if CONTINUOUS_LABEL:
        y = [window_df["event_index"] for window_df in windows_dfs]

        def window_dropping_continuous(X, y):
            def no_event_condition(y_window):
                if INCLUDE_SPO2DESAT_IN_NOEVENT:
                    return all((y_window == 0) | (y_window == 4))
                else:
                    return all(y_window == 0)

            num_of_no_event_windows = 0
            for window in y:
                if no_event_condition(window):
                    num_of_no_event_windows += 1

            num_of_event_windows = len(y) - num_of_no_event_windows
            target_num_no_event_windows = NO_EVENTS_TO_EVENTS_RATIO * num_of_event_windows
            diff = num_of_no_event_windows - target_num_no_event_windows

            # Shuffle X, y without losing order:
            combined_xy = list(zip(X, y))  # Combine X,y and y_continuous into one list
            random.shuffle(combined_xy)  # Shuffle

            if diff > 0:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                # Reduce no event windows by num_to_drop:
                combined_xy = list(
                    filterfalse(lambda Xy, c=count(): no_event_condition(Xy[1]) and next(c) < num_to_drop, combined_xy))
            elif diff < 0 and DROP_EVENT_WINDOWS_IF_NEEDED:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                # Reduce event windows by num_to_drop:
                combined_xy = list(
                    filterfalse(lambda Xy, c=count(): not no_event_condition(Xy[1]) and next(c) < num_to_drop,
                                combined_xy))
            X, y = zip(*combined_xy)  # separate Xy again
            return X, y

        if train:
            X, y = window_dropping_continuous(X, y)
    else:
        # One label per window / non-continuous
        y = [assign_window_label(window_df["event_index"]) for window_df in windows_dfs]

        def window_dropping(X, y):
            def no_event_condition(y):
                if INCLUDE_SPO2DESAT_IN_NOEVENT:
                    return (y == 0) or (y == 4)
                else:
                    return y == 0

            num_of_event_windows = np.count_nonzero(y)
            num_of_no_event_windows = len(y) - num_of_event_windows
            target_num_no_event_windows = NO_EVENTS_TO_EVENTS_RATIO * num_of_event_windows
            diff = num_of_no_event_windows - target_num_no_event_windows

            # Shuffle X, y without losing order:
            combined_xy = list(zip(X, y))  # Combine X and y into one list
            random.shuffle(combined_xy)  # Shuffle

            if diff > 0:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                # Reduce no event windows by num_to_drop:
                combined_xy = list(
                    filterfalse(lambda Xy, c=count(): no_event_condition(Xy[1]) and next(c) < num_to_drop, combined_xy))
            elif diff < 0 and DROP_EVENT_WINDOWS_IF_NEEDED:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                # Reduce event windows by num_to_drop:
                combined_xy = list(filterfalse(lambda Xy, c=count(): not no_event_condition(Xy[1])
                                                                     and next(c) < num_to_drop, combined_xy))
            X, y = zip(*combined_xy)  # separate Xy again
            return X, y

        if train:
            X, y = window_dropping(X, y)

    return X, y


def save_arrays_combined(subject_arrs_path: Path, X_train, y_train, X_test, y_test):
    """
    Saves four arrays for one subject: X_train, y_train, X_test, y_test.
    X_train has shape (num of windows in train, WINDOW_SAMPLES_SIZE, numOfSignals)
    The order of signals is Flow and then Pleth.
    y_train has shape (num of windows in train, WINDOW_SAMPLES_SIZE, 1)
    X_test has shape (num of windows in test, WINDOW_SAMPLES_SIZE, numOfSignals+1)
    y_test has shape (num of windows in test, WINDOW_SAMPLES_SIZE, 1)

    :param subject_arrs_path: Path to save subject's arrays
    :param X_train: iterable with train window signals
    :param y_train: iterable with train window labels
    :param X_test: iterable with test window signals
    :param y_test: iterable with test window signals
    :return: Nothing
    """
    # Create directory for subject:
    subject_arrs_path.mkdir(parents=True, exist_ok=True)

    if X_train is not None:
        # Transform to numpy arrays:
        X_train_arr = np.array(X_train,
                               dtype="float32")  # shape= (num of windows in train, WINDOW_SAMPLES_SIZE, numOfSignals)
        y_train_arr = np.array(y_train, dtype="uint8")  # shape= (num of windows in train, WINDOW_SAMPLES_SIZE, 1)

        # Set the paths:
        X_train_path = subject_arrs_path.joinpath("X_train")
        y_train_path = subject_arrs_path.joinpath("y_train")

        # Save the arrays:
        np.save(str(X_train_path), X_train_arr)
        np.save(str(y_train_path), y_train_arr)

    if X_test is not None:
        # Transform to numpy arrays:
        X_test_arr = np.array(X_test,
                              dtype="float32")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, numOfSignals+1)
        y_test_arr = np.array(y_test, dtype="uint8")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, 1)

        # Set the paths:
        X_test_path = subject_arrs_path.joinpath("X_test")
        y_test_path = subject_arrs_path.joinpath("y_test")

        # Save the arrays:
        np.save(str(X_test_path), X_test_arr)
        np.save(str(y_test_path), y_test_arr)


def plot_dists(train_label_counts, test_label_counts, train_label_counts_cont, test_label_counts_cont):
    print(f"Train: {train_label_counts} total={sum(train_label_counts.values())}")
    print({k: f"{100 * v / sum(train_label_counts.values()):.2f}%" for (k, v) in train_label_counts.items()})
    print(f"Test: {test_label_counts} total={sum(test_label_counts.values())}")
    print({k: f"{100 * v / sum(test_label_counts.values()):.2f}%" for (k, v) in test_label_counts.items()})
    if CONTINUOUS_LABEL:
        print(f"Train-Cont: {train_label_counts_cont} total={sum(train_label_counts_cont.values())}")
        print({k: f"{100 * v / sum(train_label_counts_cont.values()):.2f}%" for (k, v) in
               train_label_counts_cont.items()})
        print(f"Test-Cont: {test_label_counts_cont} total={sum(test_label_counts_cont.values())}")
        print(
            {k: f"{100 * v / sum(test_label_counts_cont.values()):.2f}%" for (k, v) in test_label_counts_cont.items()})

    labels = list(train_label_counts.keys())
    train_counts = list(train_label_counts.values())
    test_counts = list(test_label_counts.values())
    X_axis = np.array(labels)
    if CONTINUOUS_LABEL:
        train_counts_cont = list(train_label_counts_cont.values())
        test_counts_cont = list(test_label_counts_cont.values())

        fig_cont, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].set_title("Train set label counts")
        ax[0].bar(X_axis, train_counts_cont, 0.4, label='Sample Labels')
        ax[0].set_xticks(X_axis, [str(x) for x in labels])
        ax[0].set_xlabel("Event index")
        ax[0].set_ylabel("Count")
        ax[0].legend()

        ax[1].set_title("Test set label counts")
        ax[1].bar(X_axis, test_counts_cont, 0.4, label='Sample Labels')
        ax[1].set_xticks(X_axis, [str(x) for x in labels])
        ax[1].set_xlabel("Event index")
        ax[1].set_ylabel("Count")
        ax[1].legend()
        fig_cont.savefig(Path(PATH_TO_SUBSET).joinpath("histogram_continuous_label.png"))

        fig_win, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set_title("Train set label counts")
        ax[0].bar(X_axis, train_counts, 0.4, label='Window Labels')
        ax[0].set_xticks(X_axis, [str(x) for x in labels])
        ax[0].set_xlabel("Event index")
        ax[0].set_ylabel("Count")
        ax[0].legend()

        ax[1].set_title("Test set label counts")
        ax[1].bar(X_axis, test_counts, 0.4, label='Window Labels')
        ax[1].set_xticks(X_axis, [str(x) for x in labels])
        ax[1].set_xlabel("Event index")
        ax[1].set_ylabel("Count")
        ax[1].legend()
        fig_win.savefig(Path(PATH_TO_SUBSET).joinpath("histogram_window_label.png"))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].set_title("Train set label counts")
        ax[0].bar(X_axis, train_counts, label='Window Labels')
        ax[0].set_xticks(X_axis, [str(x) for x in labels])
        ax[0].set_xlabel("Event index")
        ax[0].set_ylabel("Count")
        ax[0].legend()

        ax[1].set_title("Test set label counts")
        ax[1].bar(X_axis, test_counts, label='Window Labels')
        ax[1].set_xticks(X_axis, [str(x) for x in labels])
        ax[1].set_xlabel("Event index")
        ax[1].set_ylabel("Count")
        ax[1].legend()

        fig.savefig(Path(PATH_TO_SUBSET).joinpath("stats", "histogram_window_label.png"))
    plt.show()


def create_arrays(ids: list[int], split_dict):
    if ids is None:
        ids, split_dict = get_ids()

    sub_seed_dict = {}
    rng_seed_pth = PATH_TO_SUBSET / "sub_seed_dict.plk"

    if rng_seed_pth.is_file():
        # Not first time generating the subset
        with open(str(rng_seed_pth), mode="rb") as file:
            sub_seed_dict = pickle.load(file)

    random.seed(SEED)  # Set the seed

    train_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    test_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    train_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    test_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for (id, sub) in get_subjects_by_ids_generator(ids, progress_bar=True):
        subject_arrs_path = Path(PATH_TO_SUBSET).joinpath("arrays", str(id).zfill(4))

        if subject_arrs_path.exists() and SKIP_EXISTING_IDS:
            continue
        else:
            subject_arrs_path.mkdir(exist_ok=True, parents=True)

        # Save metadata
        metadata = sub.metadata
        metadata_df = pd.Series(metadata)
        metadata_df.to_csv(subject_arrs_path.joinpath("sub_metadata.csv"))

        if id in sub_seed_dict:
            state = sub_seed_dict[id]
            random.setstate(state)
        else:
            # First time generating the subset
            sub_seed_dict[id] = random.getstate()
            with open(str(rng_seed_pth), mode="wb") as file:
                pickle.dump(sub_seed_dict, file)

        X_train, X_test, y_train, y_test = None, None, None, None
        if id in split_dict["train_ids"]:
            X_train, y_train = get_subject_windows(sub, train=True)
            if COUNT_LABELS and not SKIP_EXISTING_IDS:
                train_y_cont = np.array(y_train).flatten()
                train_y = y_train
                # y_cont = np.array([*y_train, *y_test]).flatten()
                # y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
                if CONTINUOUS_LABEL:
                    train_event_counts_cont = Counter(train_y_cont)  # Series containing counts of unique values.
                    train_event_counts = Counter([assign_window_label(window) for window in train_y])
                    for i in train_label_counts.keys():  # There are 5 labels
                        train_label_counts_cont[i] += train_event_counts_cont[i]
                        train_label_counts[i] += train_event_counts[i]
                else:
                    train_event_counts = Counter(train_y)  # Series containing counts of unique values.
                    for i in train_label_counts.keys():  # There are 5 labels
                        train_label_counts[i] += train_event_counts[i]

        else:
            X_test, y_test = get_subject_windows(sub, train=False)
            if COUNT_LABELS and not SKIP_EXISTING_IDS:
                test_y_cont = np.array(y_test).flatten()
                test_y = y_test
                # y_cont = np.array([*y_train, *y_test]).flatten()
                # y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
                if CONTINUOUS_LABEL:
                    test_event_counts_cont = Counter(test_y_cont)  # Series containing counts of unique values.
                    test_event_counts = Counter([assign_window_label(window) for window in test_y])
                    for i in test_label_counts.keys():  # There are 5 labels
                        test_label_counts_cont[i] += test_event_counts_cont[i]
                        test_label_counts[i] += test_event_counts[i]
                else:
                    test_event_counts = Counter(test_y)  # Series containing counts of unique values.
                    for i in test_label_counts.keys():  # There are 5 labels
                        test_label_counts[i] += test_event_counts[i]

        save_arrays_combined(subject_arrs_path, X_train, y_train, X_test, y_test)

    if COUNT_LABELS and not SKIP_EXISTING_IDS:
        plot_dists(train_label_counts, test_label_counts, train_label_counts_cont, test_label_counts_cont)


if __name__ == "__main__":
    PATH_TO_SUBSET.mkdir(exist_ok=True)
    ids, split_dict = get_ids()
    print(ids)

    if CREATE_ARRAYS:
        create_arrays(ids, split_dict)
    else:
        train_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        test_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        train_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        test_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for id in tqdm(ids):
            subject_arrs_path = Path(PATH_TO_SUBSET).joinpath("arrays", str(id).zfill(4))
            y_train_path = subject_arrs_path.joinpath("y_train.npy")
            y_test_path = subject_arrs_path.joinpath("y_test.npy")

            if y_train_path.is_file():
                y_train = np.load(str(y_train_path))
                train_y_cont = y_train.flatten()
                train_y = y_train
                if CONTINUOUS_LABEL:
                    train_event_counts_cont = Counter(train_y_cont)  # Series containing counts of unique values.
                    train_event_counts = Counter([assign_window_label(window) for window in train_y.tolist()])
                    for i in train_label_counts.keys():  # There are 5 labels
                        train_label_counts_cont[i] += train_event_counts_cont[i]
                        train_label_counts[i] += train_event_counts[i]
                else:
                    train_event_counts = Counter(train_y)  # Series containing counts of unique values.
                    for i in train_label_counts.keys():  # There are 5 labels
                        train_label_counts[i] += train_event_counts[i]
            else:
                y_test = np.load(str(y_test_path))
                test_y_cont = y_test.flatten()
                test_y = y_test
                if CONTINUOUS_LABEL:
                    test_event_counts_cont = Counter(test_y_cont)  # Series containing counts of unique values.
                    test_event_counts = Counter([assign_window_label(window) for window in test_y.tolist()])
                    for i in train_label_counts.keys():  # There are 5 labels
                        test_label_counts_cont[i] += test_event_counts_cont[i]
                        test_label_counts[i] += test_event_counts[i]
                else:
                    test_event_counts = Counter(test_y)  # Series containing counts of unique values.
                    for i in train_label_counts.keys():  # There are 5 labels
                        test_label_counts[i] += test_event_counts[i]

            # y_cont = np.array([*y_train, *y_test]).flatten()
            # y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)

        plot_dists(train_label_counts, test_label_counts, train_label_counts_cont, test_label_counts_cont)
