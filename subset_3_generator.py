# Per window labels

import math
import time
from itertools import filterfalse, count
from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# Local imports:
from common import Subject
from object_loader import all_subjects_generator, get_subjects_by_ids_generator, get_subject_by_id

# --- START OF CONSTANTS --- #
SUBSET_SIZE = 400  # The number of subjects that will remain after screening down the whole dataset
CREATE_ARRAYS = True
SKIP_EXISTING_IDS = False
WINDOW_SEC_SIZE = 16
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
STEP = 16  # The step between each window
CONTINUOUS_LABEL = False
TEST_SIZE = 0.3
TEST_SEARCH_SAMPLE_STEP = 512
EXAMINED_TEST_SETS_SUBSAMPLE = 0.7  # Ratio of randomly selected test set candidates to all possible candidates
MIN_TRAIN_WINDOW_LABEL_CONFIDENCE = 0.7
TARGET_TRAIN_TEST_SIMILARITY = 0.975  # Desired train-test similarity. 1=Identical distributions, 0=Completely different
NO_APNEA_TO_APNEA_EVENTS_RATIO = 5  # Central, Obstructive and Hypopnea are taken into account
MIN_WINDOWS = 1000  # Minimum value of subject's windows to remain after window dropping
EXCLUDE_10s_AFTER_EVENT = True
DROP_EVENT_WINDOWS_IF_NEEDED = False
EVENT_RATIO_BY_SAMPLES = True  # whether to target NO_APNEA_TO_APNEA_EVENTS_RATIO with windows or with samples
COUNT_LABELS = True
SAVE_ARRAYS_EXPANDED = False
SEED = 33

WINDOW_SAMPLES_SIZE = WINDOW_SEC_SIZE * SIGNALS_FREQUENCY

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    PATH_TO_SUBSET3 = Path(config["paths"]["local"]["subset_3_directory"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET3 = Path(__file__).parent.joinpath("data", "subset-3")


# --- END OF CONSTANTS --- #
def get_best_ids():
    path = PATH_TO_SUBSET3.joinpath("ids.npy")
    if path.is_file():
        best_ids_arr = np.load(str(path))  # array to save the best subject ids
        best_ids = best_ids_arr.tolist()  # equivalent list
    else:
        id_score_dict: dict[int: Subject] = {}  # score based on events : subject id

        # Score each subject based on events:
        for id, sub in all_subjects_generator():
            df = sub.export_to_dataframe(print_downsampling_details=False)

            # n_central_apnea_events = len(sub.get_events_by_concept("central_apnea"))
            n_central_apnea_events = df["event_index"].apply(lambda e: 1 if e == 1 else 0).sum()

            if n_central_apnea_events == 0:
                # Exclude subjects who do not have any of the desired events
                continue

            # n_obstructive_apnea_events = len(sub.get_events_by_concept("obstructive_apnea"))
            n_obstructive_apnea_events = df["event_index"].apply(lambda e: 1 if e == 2 else 0).sum()
            if n_obstructive_apnea_events == 0:
                # Exclude subjects who do not have any of the desired events
                continue

            # n_hypopnea_events = len(sub.get_events_by_concept("hypopnea"))
            n_hypopnea_events = df["event_index"].apply(lambda e: 1 if e == 3 else 0).sum()
            if n_hypopnea_events == 0:
                # Exclude subjects who do not have any of the desired events
                continue

            # n_spo2_desat_events = len(sub.get_events_by_concept("spo2_desat"))
            n_spo2_desat_events = df["event_index"].apply(lambda e: 1 if e == 4 else 0).sum()
            if n_spo2_desat_events == 0:
                # Exclude subjects who do not have any of the desired events
                continue

            # Calculate aggregate score based on most important events
            aggregate_score = n_central_apnea_events + n_obstructive_apnea_events + n_hypopnea_events
            id_score_dict[id] = aggregate_score

        # Check if threshold is met:
        if len(id_score_dict.values()) < SUBSET_SIZE:
            print(
                f"The screening criteria resulted in less than {SUBSET_SIZE} subjects ({len(id_score_dict.values())}).\n"
                f"Extra subjects will be added based on relaxed criteria")
            extra_id_score_dict = {}
            # Score each subject with relaxed standards:
            for id, sub in all_subjects_generator():
                if len(id_score_dict.values()) >= SUBSET_SIZE:
                    break
                if id not in id_score_dict.values():
                    df = sub.export_to_dataframe(print_downsampling_details=False)

                    n_central_apnea_events = df["event_index"].apply(lambda e: 1 if e == 1 else 0).sum()
                    n_obstructive_apnea_events = df["event_index"].apply(lambda e: 1 if e == 2 else 0).sum()

                    if n_central_apnea_events == 0 and n_obstructive_apnea_events == 0:
                        # Exclude subjects who do not have any of the desired events
                        continue

                    n_hypopnea_events = df["event_index"].apply(lambda e: 1 if e == 3 else 0).sum()
                    # Calculate aggregate score based on most important events
                    aggregate_score = n_central_apnea_events + n_obstructive_apnea_events + n_hypopnea_events
                    extra_id_score_dict[id] = aggregate_score

            print(
                f"Screened dataset (strict + relaxed criteria) size: {len(id_score_dict.keys()) + len(extra_id_score_dict.keys())}")
            # Rank the extra relaxed screened subjects that will supplement the strictly screened subjects:
            top_screened_ids = sorted(extra_id_score_dict.keys(), key=lambda id: extra_id_score_dict[id], reverse=True
                                      )[0:(SUBSET_SIZE - len(id_score_dict.keys()))]
            best_ids = list(id_score_dict.keys())  # Add all the strictly screened subjects
            best_ids.extend(top_screened_ids)  # Add the extra ranked subjects
        else:
            print(f"Screened dataset size: {len(id_score_dict.keys())}")
            # List with top 400 ids:
            top_screened_ids = sorted(id_score_dict.keys(), key=lambda id: id_score_dict[id], reverse=True)[
                               0:SUBSET_SIZE]
            best_ids = top_screened_ids  # List with top 400 ids

        best_ids_arr = np.array(best_ids)  # Equivalent array
        path = PATH_TO_SUBSET3.joinpath("ids")
        np.save(str(path), best_ids_arr)
        path = PATH_TO_SUBSET3.joinpath("ids.csv")
        best_ids_arr.tofile(str(path), sep=',')
    return best_ids.copy()


# best_ids = get_best_ids()
# print(f"Final subset size: {len(best_ids)}\n")
# print(best_ids)


def assign_window_label(window_labels: pd.Series) -> tuple[int, float]:
    """
    :param window_labels: A pandas Series with the event labels for each sample in the window
    :return: One label representing the whole window, specifically the label with most occurrences.
    """
    if not isinstance(window_labels, pd.Series):
        window_labels = pd.Series(window_labels)

    # 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation
    event_counts = window_labels.value_counts()  # Series containing counts of unique values.
    prominent_event_index = event_counts.argmax()
    prominent_event = event_counts.index[prominent_event_index]
    confidence = event_counts[prominent_event] / WINDOW_SAMPLES_SIZE
    # print(event_counts)
    return int(prominent_event), float(confidence)


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


def get_subject_train_test_data(subject: Subject, sufficiently_low_divergence=None) \
        -> tuple[list, list, list[pd.Series] | list[int], list[pd.Series] | list[int]]:
    sub_df = subject.export_to_dataframe(signal_labels=["Flow", "Pleth"], print_downsampling_details=False, anti_aliasing=False)
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
        # print(f"i={i}/{num_of_candidates}")
        # print(f"JSD: {divergence:.4f}")
        # y_train = train_df["event_index"]
        # y_test = test_df["event_index"]
        # print(f"TRAIN  -  CA: {sum(y_train == 1) / len(y_train):.2f}  OA: {sum(y_train == 2) / len(y_train):.2f}  "
        #       f"H: {sum(y_train == 3) / len(y_train):.2f}  S: {sum(y_train == 4) / len(y_train):.2f}")
        # print(f"TEST  -  CA: {sum(y_test == 1) / len(y_test):.2f}  OA: {sum(y_test == 2) / len(y_test):.2f}  "
        #       f"H: {sum(y_test == 3) / len(y_test):.2f}  S: {sum(y_test == 4) / len(y_test):.2f}")
        # print("-------------------------------------------------------------------------------------------------------")

        # We want to minimize the divergence because we want to maximize similarity
        if divergence < min_split_divergence:
            min_split_divergence = divergence
            best_split = (train_df, test_df)
            if divergence < sufficiently_low_divergence:
                break

    train_df = best_split[0]
    test_df = best_split[1]

    samples_to_exclude = []
    if EXCLUDE_10s_AFTER_EVENT:
        # 2. Drop no-event train samples within 10s (= 320 samples) after an event:

        # for every sample
        for i in train_df.index:
            # if it has event:
            if train_df.loc[i, "event_index"] != 0:
                # Check the next 10s for no event samples:
                blacklist_tmp = []
                for j in range(i + 1, i + 10*SIGNALS_FREQUENCY+1):
                    # Check if continuity breaks at j (due to train test split):
                    if j not in train_df.index:
                        # if j is not in index it means that train is not continuous everywhere because of the train
                        # test split. Consequently, train is continuous on two parts (one part before and one part after
                        # the test). This means that j-1 was the last point of the first part.
                        break
                    if train_df.loc[j, "event_index"] == 0:
                        # Add no event sample to blacklist:
                        blacklist_tmp.append(j)
                    else:
                        # Event sample found within 10s of the event sample that we examine,
                        # thus there is to examine for more no event samples, since i will reach that event.
                        # Also, the blacklisting of these samples will not be enforced in order to keep this event.
                        blacklist_tmp.clear()
                        break
                samples_to_exclude.extend(blacklist_tmp)
    # print(samples_to_exclude)

    # Take equal-sized windows with a specified step:
    # 3. Calculate the number of windows:
    num_windows_train = (len(train_df) - WINDOW_SAMPLES_SIZE) // STEP + 1  # a//b = math.floor(a/b)
    num_windows_test = (len(test_df) - WINDOW_SAMPLES_SIZE) // STEP + 1  # a//b = math.floor(a/b)
    # Note that due to floor division the last WINDOW_SAMPLES_SIZE-1 samples might be dropped

    # 4. Generate equal-sized windows:
    train_windows_dfs = list()
    train_index = train_df.index

    for i in range(num_windows_train):
        start = train_index[i * STEP]
        stop = train_index[i * STEP + WINDOW_SAMPLES_SIZE - 1]

        # Check continuity:
        if (start + WINDOW_SAMPLES_SIZE - 1) != stop:
            # if true it means that train is not continuous everywhere because of the train test split.
            # Consequently, train is continuous on two parts (one part before and one part after
            # the test). Window has to be cut from continuous region, so we skip this.
            continue

        window_df = train_df.loc[start:stop, :]  # Note that contrary to usual slices, both start and stop are included

        # Check if it does not contain events
        if not any(window_df["event_index"] != 0):
            # It is pure no event window, check if it should be excluded.
            if start in samples_to_exclude:
                # Window definitely contains samples marked for exclusion.
                # Exclusion zone is continuous and always has an event one sample before. As a result checking
                # the start is adequate for determining if window contains marked samples because window contains
                # only no events.
                continue

        train_windows_dfs.append(window_df)

    # train_windows_dfs = [train_df.iloc[i * STEP:i * STEP + WINDOW_SAMPLES_SIZE] for i in range(num_windows_train)]
    test_windows_dfs = [test_df.iloc[i * STEP:i * STEP + WINDOW_SAMPLES_SIZE] for i in range(num_windows_test)]
    # Note that when using df.iloc[] or df[], the stop part is not included. However ,when using loc stop is included

    X_train = [window_df.loc[:, window_df.columns != "event_index"] for window_df in train_windows_dfs]
    X_test = [window_df.loc[:, window_df.columns != "event_index"] for window_df in test_windows_dfs]

    # 5. Drop no-event windows to achieve desired ratio of no-events to events:
    if CONTINUOUS_LABEL:
        y_train = [window_df["event_index"] for window_df in train_windows_dfs]
        y_test = [window_df["event_index"] for window_df in test_windows_dfs]

        def window_dropping_continuous(X, y):
            if not EVENT_RATIO_BY_SAMPLES:
                num_of_apnea_windows = sum([any((window >= 1) & (window <= 3)) for window in y])
                num_of_no_apnea_windows = len(y) - num_of_apnea_windows
                target_num_no_apnea_windows = NO_APNEA_TO_APNEA_EVENTS_RATIO * num_of_apnea_windows
                diff = num_of_no_apnea_windows - target_num_no_apnea_windows
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
            else:
                total_samples = sum([len(window) for window in y])
                num_of_apnea_samples = sum([sum((window >= 1) & (window <= 3)) for window in y])
                num_of_no_apnea_samples = total_samples - num_of_apnea_samples
                target_num_no_apnea_samples = NO_APNEA_TO_APNEA_EVENTS_RATIO * num_of_apnea_samples
                diff = num_of_no_apnea_samples - target_num_no_apnea_samples
                num_to_drop = min(abs(diff), total_samples - MIN_WINDOWS * WINDOW_SAMPLES_SIZE)

            # Shuffle X, y without losing order:
            combined_xy = list(zip(X, y))  # Combine X,y and y_continuous into one list
            random.shuffle(combined_xy)  # Shuffle

            if diff > 0:
                if not EVENT_RATIO_BY_SAMPLES:
                    # Reduce no event windows by num_to_drop:
                    combined_xy = list(
                        filterfalse(lambda Xy, c=count(): all((Xy[1] == 0) | (Xy[1] == 4))
                                                          and next(c) < num_to_drop, combined_xy))
                else:
                    samples_dropped = 0
                    windows_to_drop = []
                    for window_i in range(len(combined_xy)):
                        if samples_dropped > num_to_drop:
                            break
                        else:
                            xy = combined_xy[window_i]
                            # Check if it contains only normal and spo2 desat
                            if all((xy[1] == 0) | (xy[1] == 4)):
                                # To be dropped
                                windows_to_drop.append(window_i)
                                samples_dropped += len(xy[1])
                    combined_xy = [combined_xy[i] for i in range(len(combined_xy)) if i not in windows_to_drop]

            elif diff < 0 and DROP_EVENT_WINDOWS_IF_NEEDED:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                if not EVENT_RATIO_BY_SAMPLES:
                    # Reduce event windows by num_to_drop:
                    combined_xy = list(
                        filterfalse(lambda Xy, c=count(): all((Xy[1] >= 1) & (Xy[1] <= 3)) and next(c) < num_to_drop,
                                    combined_xy))
                else:
                    samples_dropped = 0
                    windows_to_drop = []
                    for window_i in range(len(combined_xy)):
                        if samples_dropped > num_to_drop:
                            break
                        else:
                            xy = combined_xy[window_i]
                            # Check if it contains apnea
                            if all((xy[1] >= 1) | (xy[1] <= 3)):
                                # To be dropped
                                windows_to_drop.append(window_i)
                                samples_dropped += len(xy[1])
                    combined_xy = [combined_xy[i] for i in range(len(combined_xy)) if i not in windows_to_drop]
            X, y = zip(*combined_xy)  # separate Xy again
            return X, y

        X_train, y_train = window_dropping_continuous(X_train, y_train)
        # X_test, y_test = window_dropping_continuous(X_test, y_test)
    else:
        # One label per window / non-continuous
        y_train = [assign_window_label(window_df["event_index"])[0] for window_df in train_windows_dfs
                   if assign_window_label(window_df["event_index"])[1] >= MIN_TRAIN_WINDOW_LABEL_CONFIDENCE]
        y_test = [assign_window_label(window_df["event_index"])[0] for window_df in test_windows_dfs]

        def window_dropping(X, y):
            num_of_apnea_windows = len([yi for yi in y if (yi >= 1) and (yi <= 3)])
            num_of_no_apnea_windows = len(y) - num_of_apnea_windows

            target_num_no_apnea_windows = NO_APNEA_TO_APNEA_EVENTS_RATIO * num_of_apnea_windows
            diff = num_of_no_apnea_windows - target_num_no_apnea_windows

            # Shuffle X, y without losing order:
            combined_xy = list(zip(X, y))  # Combine X and y into one list
            random.shuffle(combined_xy)  # Shuffle

            if diff > 0:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                # Reduce no event windows by num_to_drop:
                combined_xy = list(filterfalse(lambda Xy, c=count(): (Xy[1] == 0 or Xy[1] == 4)
                                                                     and next(c) < num_to_drop, combined_xy))
            elif diff < 0 and DROP_EVENT_WINDOWS_IF_NEEDED:
                num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
                # Reduce event windows by num_to_drop:
                combined_xy = list(filterfalse(lambda Xy, c=count(): Xy[1] != 0 and next(c) < num_to_drop, combined_xy))
            X, y = zip(*combined_xy)  # separate Xy again
            return X, y

        X_train, y_train = window_dropping(X_train, y_train)
        # X_test, y_test = window_dropping(X_test, y_test)
    return X_train, X_test, y_train, y_test


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
    # Transform to numpy arrays:
    X_train_arr = np.array(X_train,
                           dtype="float32")  # shape= (num of windows in train, WINDOW_SAMPLES_SIZE, numOfSignals)
    y_train_arr = np.array(y_train, dtype="uint8")  # shape= (num of windows in train, WINDOW_SAMPLES_SIZE, 1)
    X_test_arr = np.array(X_test,
                          dtype="float32")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, numOfSignals+1)
    y_test_arr = np.array(y_test, dtype="uint8")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, 1)

    # Create directory for subject:
    subject_arrs_path.mkdir(parents=True, exist_ok=True)

    X_train_path = subject_arrs_path.joinpath("X_train")
    y_train_path = subject_arrs_path.joinpath("y_train")
    X_test_path = subject_arrs_path.joinpath("X_test")
    y_test_path = subject_arrs_path.joinpath("y_test")

    # Save the arrays
    np.save(str(X_train_path), X_train_arr)
    np.save(str(y_train_path), y_train_arr)
    np.save(str(X_test_path), X_test_arr)
    np.save(str(y_test_path), y_test_arr)


def save_arrays_expanded(subject_arrs_path: Path, X_train, y_train, X_test, y_test):
    """
    Saves one array per window for one subject in two directories: train, test.
    X_{index} has shape (WINDOW_SAMPLES_SIZE, numOfSignals)
    y_{index} has shape (WINDOW_SAMPLES_SIZE, 1)

    :param subject_arrs_path: Path to save subject's arrays
    :param X_train: iterable with train window signals
    :param y_train: iterable with train window labels
    :param X_test: iterable with test window signals
    :param y_test: iterable with test window signals
    :return: Nothing
    """
    # Save train arrays, one file each
    n_train_windows = len(y_train)
    for w in range(n_train_windows):
        # Transform window to numpy array:
        X_window = np.array(X_train[w], dtype="float32").reshape(WINDOW_SAMPLES_SIZE, -1)
        y_window = np.array(y_train[w], dtype="uint8").ravel()

        # Create directory for subject:
        subject_train_dir = subject_arrs_path.joinpath(str(id).zfill(4), "train")
        subject_train_dir.mkdir(parents=True, exist_ok=True)

        X_window_path = subject_train_dir.joinpath(f"X_{w}.npy")
        y_window_path = subject_train_dir.joinpath(f"y_{w}.npy")

        # Save the arrays
        np.save(str(X_window_path), X_window)
        np.save(str(y_window_path), y_window)

    # Same for test:
    n_test_windows = len(y_test)
    for w in range(n_test_windows):
        # Transform window to numpy array:
        X_window = np.array(X_test[w], dtype="float32").reshape(WINDOW_SAMPLES_SIZE, -1)
        y_window = np.array(y_test[w], dtype="uint8").ravel()

        # Create directory for subject:
        subject_test_dir = subject_arrs_path.joinpath(str(id).zfill(4), "test")
        subject_test_dir.mkdir(parents=True, exist_ok=True)

        X_window_path = subject_test_dir.joinpath(f"X_{w}.npy")
        y_window_path = subject_test_dir.joinpath(f"y_{w}.npy")

        # Save the arrays
        np.save(str(X_window_path), X_window)
        np.save(str(y_window_path), y_window)


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

        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].set_title("Train set label counts")
        ax[0].bar(X_axis - 0.2, train_counts, 0.4, label='Window Labels')
        ax[0].bar(X_axis + 0.2, train_counts_cont, 0.4, label='Sample Labels')
        ax[0].set_xticks(X_axis, [str(x) for x in labels])
        ax[0].set_xlabel("Event index")
        ax[0].set_ylabel("Count")
        ax[0].legend()

        ax[1].set_title("Test set label counts")
        ax[1].bar(X_axis - 0.2, test_counts, 0.4, label='Window Labels')
        ax[1].bar(X_axis + 0.2, test_counts_cont, 0.4, label='Sample Labels')
        ax[1].set_xticks(X_axis, [str(x) for x in labels])
        ax[1].set_xlabel("Event index")
        ax[1].set_ylabel("Count")
        ax[1].legend()

        fig.savefig(Path(PATH_TO_SUBSET3).joinpath("histogram_continuous_label.png"))
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

        fig.savefig(Path(PATH_TO_SUBSET3).joinpath("histogram_window_label.png"))
    plt.show()


def create_arrays(ids: list[int]):
    if ids is None:
        ids = get_best_ids()

    train_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    test_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    train_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    test_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for (id, sub) in get_subjects_by_ids_generator(ids, progress_bar=True):
        random.seed(SEED)  # Set the seed

        subject_arrs_path = PATH_TO_SUBSET3.joinpath("arrays", str(id).zfill(4))

        if subject_arrs_path.exists() and SKIP_EXISTING_IDS:
            continue
        else:
            subject_arrs_path.mkdir(exist_ok=True, parents=True)
        # Save metadata
        metadata = sub.metadata
        metadata_df = pd.Series(metadata)
        metadata_df.to_csv(subject_arrs_path.joinpath("sub_metadata.csv"))

        X_train, X_test, y_train, y_test = get_subject_train_test_data(sub)

        if COUNT_LABELS and not SKIP_EXISTING_IDS:
            train_y_cont = np.array(y_train).flatten()
            test_y_cont = np.array(y_test).flatten()
            train_y = y_train
            test_y = y_test
            # y_cont = np.array([*y_train, *y_test]).flatten()
            # y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
            if CONTINUOUS_LABEL:
                train_event_counts_cont = Counter(train_y_cont)  # Series containing counts of unique values.
                test_event_counts_cont = Counter(test_y_cont)  # Series containing counts of unique values.
                train_event_counts = Counter([assign_window_label(window)[0] for window in train_y])
                test_event_counts = Counter([assign_window_label(window)[0] for window in test_y])
                for i in train_label_counts.keys():  # There are 5 labels
                    train_label_counts_cont[i] += train_event_counts_cont[i]
                    test_label_counts_cont[i] += test_event_counts_cont[i]
                    train_label_counts[i] += train_event_counts[i]
                    test_label_counts[i] += test_event_counts[i]
            else:
                train_event_counts = Counter(train_y)  # Series containing counts of unique values.
                test_event_counts = Counter(test_y)  # Series containing counts of unique values.
                for i in train_label_counts.keys():  # There are 5 labels
                    train_label_counts[i] += train_event_counts[i]
                    test_label_counts[i] += test_event_counts[i]

        if SAVE_ARRAYS_EXPANDED:
            save_arrays_expanded(subject_arrs_path, X_train, y_train, X_test, y_test)
        else:
            save_arrays_combined(subject_arrs_path, X_train, y_train, X_test, y_test)
    plot_dists(train_label_counts, test_label_counts, train_label_counts_cont, test_label_counts_cont)


if __name__ == "__main__":
    PATH_TO_SUBSET3.mkdir(exist_ok=True)
    best_ids = get_best_ids()
    print(best_ids)

    if CREATE_ARRAYS:
        create_arrays(best_ids)
    else:
        train_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        test_label_counts: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        train_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        test_label_counts_cont: dict[int: int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for id in tqdm(best_ids):
            subject_arrs_path = Path(PATH_TO_SUBSET3).joinpath("arrays", str(id).zfill(4))
            y_train_path = subject_arrs_path.joinpath("y_train.npy")
            y_test_path = subject_arrs_path.joinpath("y_test.npy")

            y_train = np.load(str(y_train_path))
            y_test = np.load(str(y_test_path))

            train_y_cont = y_train.flatten()
            test_y_cont = y_test.flatten()
            train_y = y_train
            test_y = y_test
            # y_cont = np.array([*y_train, *y_test]).flatten()
            # y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
            if CONTINUOUS_LABEL:
                train_event_counts_cont = Counter(train_y_cont)  # Series containing counts of unique values.
                test_event_counts_cont = Counter(test_y_cont)  # Series containing counts of unique values.
                train_event_counts = Counter([assign_window_label(window)[0] for window in train_y.tolist()])
                test_event_counts = Counter([assign_window_label(window)[0] for window in test_y.tolist()])
                for i in train_label_counts.keys():  # There are 5 labels
                    train_label_counts_cont[i] += train_event_counts_cont[i]
                    test_label_counts_cont[i] += test_event_counts_cont[i]
                    train_label_counts[i] += train_event_counts[i]
                    test_label_counts[i] += test_event_counts[i]
            else:
                train_event_counts = Counter(train_y)  # Series containing counts of unique values.
                test_event_counts = Counter(test_y)  # Series containing counts of unique values.
                for i in train_label_counts.keys():  # There are 5 labels
                    train_label_counts[i] += train_event_counts[i]
                    test_label_counts[i] += test_event_counts[i]
        plot_dists(train_label_counts, test_label_counts, train_label_counts_cont, test_label_counts_cont)
