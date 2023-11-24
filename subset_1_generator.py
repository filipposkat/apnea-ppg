from itertools import filterfalse, count
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random
from sklearn.model_selection import train_test_split

# Local imports:
from common import Subject
from object_loader import all_subjects_generator, get_subjects_by_ids_generator

# --- START OF CONSTANTS --- #
SUBSET_SIZE = 400  # The number of subjects that will remain after screening down the whole dataset
WINDOW_SEC_SIZE = 16
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
STEP = 2  # The step between each window
CONTINUOUS_LABEL = True
TEST_SIZE = 0.2
NO_EVENTS_TO_EVENTS_RATIO = 5
MIN_WINDOWS = 1000  # Minimum value of subject's windows to remain after window dropping
DROP_EVENT_WINDOWS_IF_NEEDED = False
CREATE_ARRAYS = True
SKIP_EXISTING_IDS = False
COUNT_LABELS = True
SEED = 33

WINDOW_SAMPLES_SIZE = WINDOW_SEC_SIZE * SIGNALS_FREQUENCY

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = config["paths"]["local"]["subject_objects_directory"]
    PATH_TO_SUBSET1 = config["paths"]["local"]["subset_1_directory"]
else:
    PATH_TO_OBJECTS = Path(__file__).joinpath("data", "serialized-objects")
    PATH_TO_SUBSET1 = Path(__file__).joinpath("data", "subset-1")
# --- END OF CONSTANTS --- #

path = Path(PATH_TO_SUBSET1).joinpath("ids.npy")
if path.is_file():
    best_ids_arr = np.load(str(path))  # array to save the best subject ids
    best_ids = best_ids_arr.tolist()  # equivalent list
else:
    score_id_dict: dict[int: Subject] = {}  # score based on events : subject id

    # Score each subject based on events:
    for id, sub in all_subjects_generator():
        n_central_apnea_events = len(sub.get_events_by_concept("central_apnea"))
        if n_central_apnea_events == 0:
            # Exclude subjects who do not have any of the desired events
            continue

        n_obstructive_apnea_events = len(sub.get_events_by_concept("obstructive_apnea"))
        if n_obstructive_apnea_events == 0:
            # Exclude subjects who do not have any of the desired events
            continue

        n_hypopnea_events = len(sub.get_events_by_concept("hypopnea"))
        if n_hypopnea_events == 0:
            # Exclude subjects who do not have any of the desired events
            continue

        n_spo2_desat_events = len(sub.get_events_by_concept("spo2_desat"))
        if n_spo2_desat_events == 0:
            # Exclude subjects who do not have any of the desired events
            continue

        # Calculate aggregate score based on most important events
        aggregate_score = n_central_apnea_events + n_obstructive_apnea_events + n_hypopnea_events
        score_id_dict[aggregate_score] = id

    # Check if threshold is met:
    if len(score_id_dict.values()) < SUBSET_SIZE:
        print(f"The screening criteria resulted in less than {SUBSET_SIZE} subjects ({len(score_id_dict.values())}).\n"
              f"Extra subjects will be added based on relaxed criteria")
        extra_score_id_dict = {}
        # Score each subject with relaxed standards:
        for id, sub in all_subjects_generator():
            if len(score_id_dict.values()) >= SUBSET_SIZE:
                break
            if id not in score_id_dict.values():
                n_central_apnea_events = len(sub.get_events_by_concept("central_apnea"))
                n_obstructive_apnea_events = len(sub.get_events_by_concept("obstructive_apnea"))
                if n_central_apnea_events == 0 and n_obstructive_apnea_events == 0:
                    # Exclude subjects who do not have any of the desired events
                    continue
                n_hypopnea_events = len(sub.get_events_by_concept("hypopnea"))
                # Calculate aggregate score based on most important events
                aggregate_score = n_central_apnea_events + n_obstructive_apnea_events + n_hypopnea_events
                extra_score_id_dict[aggregate_score] = id

        print(f"Screened dataset size: {len(score_id_dict.values()) + len(extra_score_id_dict)}")
        # Rank the extra relaxed screened subjects that will supplement the strictly screened subjects:
        top_screened_scores = sorted(extra_score_id_dict, reverse=True)[0:(SUBSET_SIZE - len(score_id_dict.values()))]
        best_ids = list(score_id_dict.values())  # Add all the strictly screened subjects
        best_ids.extend([extra_score_id_dict[score] for score in top_screened_scores])  # Add the extra ranked subjects
    else:
        print(f"Screened dataset size: {len(score_id_dict.values())}")
        top_screened_scores = sorted(score_id_dict, reverse=True)[0:SUBSET_SIZE]  # List with top 400 scores
        best_ids = [score_id_dict[score] for score in top_screened_scores]  # List with top 400 ids

    best_ids_arr = np.array(best_ids)  # Equivalent array
    path = Path(PATH_TO_SUBSET1).joinpath("ids")
    np.save(str(path), best_ids_arr)

print(f"Final subset size: {len(best_ids)}\n")
print(best_ids)


def assign_window_label(window_labels: pd.Series) -> int:
    """
    :param window_labels: A pandas Series with the event labels for each sample in the window
    :return: One label representing the whole window, specifically the label with most occurrences.
    """
    if window_labels.max() == 0:
        return 0
    else:
        # 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation, 5=other event
        event_counts = window_labels.value_counts()  # Series containing counts of unique values.
        prominent_event_index = event_counts.argmax()
        prominent_event = event_counts.index[prominent_event_index]
        # print(event_counts)
        return int(prominent_event)


def get_subject_train_test_data(subject: Subject):
    sub_df = subject.export_to_dataframe(signal_labels=["Flow", "Pleth"], print_downsampling_details=False)
    sub_df.drop(["time_secs"], axis=1, inplace=True)

    # 1. Drop no-event samples within 10s (= 320 samples) after an event:
    samples_to_drop = []
    # for every window:
    for i in range(len(sub_df["event_index"])):
        # if it is an event window:
        if sub_df.loc[i, "event_index"] != 0:
            # Check the next 10s for no event windows:
            for j in range(i + 1, 320):
                if sub_df.loc[j, "event_index"] == 0:
                    # Add no event window to drop list:
                    samples_to_drop.append(j)
                else:
                    # Event window found within 10s of the event window we examine,
                    # thus there is no need to drop more no event windows since i will reach this event
                    break
    sub_df.drop(samples_to_drop, axis=0, inplace=True)  # Drop marked samples

    # Take equal-sized windows with a specified step:
    # 2. Calculate the number of windows:
    num_windows = (len(sub_df) - WINDOW_SAMPLES_SIZE) // STEP + 1  # a//b = math.floor(a/b)
    # Note that due to floor division the last WINDOW_SAMPLES_SIZE-1 samples might be dropped

    # 3. Generate equal-sized windows:
    windows_dfs = [sub_df.iloc[i * STEP:i * STEP + WINDOW_SAMPLES_SIZE] for i in range(num_windows)]
    X = [window_df.loc[:, window_df.columns != "event_index"] for window_df in windows_dfs]
    y = [assign_window_label(window_df["event_index"]) for window_df in windows_dfs]
    if CONTINUOUS_LABEL:
        y_cont = [window_df["event_index"] for window_df in windows_dfs]
    else:
        y_cont = list(np.zeros(len(y)))

    # # 3. Drop no-event windows within 10s (= 320 samples) after event window:
    # drop_buffer = (10 * 32 - WINDOW_SAMPLES_SIZE) // STEP + 1
    # windows_to_drop = []
    # # for every window:
    # for i in range(len(y)):
    #     # if it is an event window:
    #     if y[i] != 0:
    #         # Check the next 10s for no event windows:
    #         for j in range(i + 1, i + drop_buffer):
    #             if y[j] == 0:
    #                 # Add no event window to drop list:
    #                 windows_to_drop.append(j)
    #             else:
    #                 # Event window found within 10s of the event window we examine,
    #                 # thus there is no need to drop more no event windows since i will reach this event
    #                 break

    # 4. Drop no-event windows to achieve desired ratio of no-events to events:
    num_of_event_windows = np.count_nonzero(y)
    num_of_no_event_windows = len(y) - num_of_event_windows
    target_num_no_event_windows = NO_EVENTS_TO_EVENTS_RATIO * num_of_event_windows
    diff = num_of_no_event_windows - target_num_no_event_windows

    # Shuffle X, y without losing order:
    combined_xy = list(zip(X, y, y_cont))  # Combine X,y and y_continuous into one list
    random.shuffle(combined_xy)  # Shuffle

    if diff > 0:
        num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
        # Reduce no event windows by num_to_drop:
        combined_xy = list(filterfalse(lambda Xy, c=count(): Xy[1] == 0 and next(c) < num_to_drop, combined_xy))
    elif diff < 0 and DROP_EVENT_WINDOWS_IF_NEEDED:
        num_to_drop = min(abs(diff), (len(y) - MIN_WINDOWS))
        # Reduce event windows by num_to_drop:
        combined_xy = list(filterfalse(lambda Xy, c=count(): Xy[1] != 0 and next(c) < num_to_drop, combined_xy))
    X, y, y_cont = zip(*combined_xy)  # separate Xy again

    # 5. Split into train and test:
    if CONTINUOUS_LABEL:
        X_train, X_test, y_train, y_test = train_test_split(X, y_cont, test_size=TEST_SIZE, shuffle=True, stratify=y,
                                                            random_state=SEED)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, stratify=y,
                                                            random_state=SEED)
    # print(f"Events train%: {np.count_nonzero(y_train) / len(y_train)}")
    # print(f"Events test%: {np.count_nonzero(y_test) / len(y_test)}")
    return X_train, X_test, y_train, y_test


if CREATE_ARRAYS:
    random.seed(SEED)  # Set the seed

    label_counts: dict[int: list[int]] = dict()
    for (id, sub) in get_subjects_by_ids_generator(best_ids[0:2], progress_bar=True):
        subject_arrs_path = Path(PATH_TO_SUBSET1).joinpath("arrays", str(id).zfill(4))

        if subject_arrs_path.exists() and SKIP_EXISTING_IDS:
            continue

        X_train, X_test, y_train, y_test = get_subject_train_test_data(sub)

        if COUNT_LABELS and not SKIP_EXISTING_IDS:
            y = pd.concat([y_train, y_test])
            event_counts = y.value_counts()  # Series containing counts of unique values.
            for i in range(6):  # There are 6 labels
                if float(i) in event_counts.index:
                    label_counts[i] += event_counts.at[float(i)]

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

    histogram = sn.histplot(list(label_counts.keys()), weights=list(label_counts.values()))
    if CONTINUOUS_LABEL:
        plt.savefig(Path(PATH_TO_SUBSET1).joinpath("histogram_continuous_label.png"))
    else:
        plt.savefig(Path(PATH_TO_SUBSET1).joinpath("histogram_window_label.png"))
    plt.show()
