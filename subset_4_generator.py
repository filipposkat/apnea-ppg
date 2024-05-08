# Subset with low apnea events

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
CREATE_ARRAYS = False
SKIP_EXISTING_IDS = False
WINDOW_SEC_SIZE = 60  # -> 60 * 32 = 4096
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
STEP = 64  # The step between each window
CONTINUOUS_LABEL = True
TEST_SIZE = 0.3
TEST_SEARCH_SAMPLE_STEP = 4096
EXAMINED_TEST_SETS_SUBSAMPLE = 0.7  # Ratio of randomly selected test set candidates to all possible candidates
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
    PATH_TO_SUBSET = Path(config["paths"]["local"]["subset_4_directory"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-4")


# --- END OF CONSTANTS --- #
def get_best_ids():
    path = PATH_TO_SUBSET.joinpath("ids.npy")
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

            # if n_central_apnea_events == 0:
            #     # Exclude subjects who do not have any of the desired events
            #     continue

            # n_obstructive_apnea_events = len(sub.get_events_by_concept("obstructive_apnea"))
            n_obstructive_apnea_events = df["event_index"].apply(lambda e: 1 if e == 2 else 0).sum()
            # if n_obstructive_apnea_events == 0:
            #     # Exclude subjects who do not have any of the desired events
            #     continue

            # n_hypopnea_events = len(sub.get_events_by_concept("hypopnea"))
            n_hypopnea_events = df["event_index"].apply(lambda e: 1 if e == 3 else 0).sum()
            # if n_hypopnea_events == 0:
            #     # Exclude subjects who do not have any of the desired events
            #     continue

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
            top_screened_ids = sorted(extra_id_score_dict.keys(), key=lambda id: extra_id_score_dict[id], reverse=False
                                      )[0:(SUBSET_SIZE - len(id_score_dict.keys()))]
            best_ids = list(id_score_dict.keys())  # Add all the strictly screened subjects
            best_ids.extend(top_screened_ids)  # Add the extra ranked subjects
        else:
            print(f"Screened dataset size: {len(id_score_dict.keys())}")
            # List with top ids:
            top_screened_ids = sorted(id_score_dict.keys(), key=lambda id: id_score_dict[id], reverse=False)[
                               0:SUBSET_SIZE]
            best_ids = top_screened_ids  # List with top 400 ids

        best_ids_arr = np.array(best_ids)  # Equivalent array
        path = PATH_TO_SUBSET.joinpath("ids")
        np.save(str(path), best_ids_arr)
        path = PATH_TO_SUBSET.joinpath("ids.csv")
        best_ids_arr.tofile(str(path), sep=',')
    return best_ids.copy()


if __name__ == "__main__":
    PATH_TO_SUBSET.mkdir(exist_ok=True)
    best_ids = get_best_ids()
    print(f"Final subset size: {len(best_ids)}\n")
    print(sorted(best_ids))

