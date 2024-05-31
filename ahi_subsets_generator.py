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
    PATH_TO_SUBSET_NO = Path(config["paths"]["local"]["subset_no_directory"])
    PATH_TO_SUBSET_MILD = Path(config["paths"]["local"]["subset_mild_directory"])
    PATH_TO_SUBSET_MODERATE = Path(config["paths"]["local"]["subset_moderate_directory"])
    PATH_TO_SUBSET_SEVERE = Path(config["paths"]["local"]["subset_severe_directory"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET_NO = Path(__file__).parent.joinpath("data", "subset-no")
    PATH_TO_SUBSET_MILD = Path(__file__).parent.joinpath("data", "subset-mild")
    PATH_TO_SUBSET_MODERATE = Path(__file__).parent.joinpath("data", "subset-moderate")
    PATH_TO_SUBSET_SEVERE = Path(__file__).parent.joinpath("data", "subset-severe")


# --- END OF CONSTANTS --- #
def find_ids() -> tuple[list[int], list[int], list[int], list[int]]:
    path_no = PATH_TO_SUBSET_NO.joinpath("ids.npy")
    path_mild = PATH_TO_SUBSET_MILD.joinpath("ids.npy")
    path_moderate = PATH_TO_SUBSET_MODERATE.joinpath("ids.npy")
    path_severe = PATH_TO_SUBSET_SEVERE.joinpath("ids.npy")

    if path_no.is_file() and path_mild.is_file() and path_moderate.is_file() and path_severe.is_file():
        no_ah_ids_arr = np.load(str(path_no))
        mild_ah_ids_arr = np.load(str(path_mild))
        moderate_ah_ids_arr = np.load(str(path_moderate))
        severe_ah_ids_arr = np.load(str(path_severe))

        no_ah_ids = no_ah_ids_arr.tolist()
        mild_ah_ids = mild_ah_ids_arr.tolist()
        moderate_ah_ids = moderate_ah_ids_arr.tolist()
        severe_ah_ids = severe_ah_ids_arr.tolist()
    else:
        no_ah_ids = []
        mild_ah_ids = []
        moderate_ah_ids = []
        severe_ah_ids = []

        # Score each subject based on events:
        for id, sub in all_subjects_generator():
            metadata = sub.metadata
            ahi_a0h3a = float(metadata["ahi_a0h3a"])

            if ahi_a0h3a < 5:
                # No
                no_ah_ids.append(id)
            elif ahi_a0h3a < 15:
                # Mild
                mild_ah_ids.append(id)
            elif ahi_a0h3a < 30:
                # Moderate
                moderate_ah_ids.append(id)
            else:
                # Severe
                severe_ah_ids.append(id)

        no_ah_ids_arr = np.array(no_ah_ids)
        mild_ah_ids_arr = np.array(mild_ah_ids)
        moderate_ah_ids_arr = np.array(moderate_ah_ids)
        severe_ah_ids_arr = np.array(severe_ah_ids)

        np.save(str(path_no), no_ah_ids_arr)
        np.save(str(path_mild), mild_ah_ids_arr)
        np.save(str(path_moderate), moderate_ah_ids_arr)
        np.save(str(path_severe), severe_ah_ids_arr)

        path_no = PATH_TO_SUBSET_NO.joinpath("ids.csv")
        path_mild = PATH_TO_SUBSET_MILD.joinpath("ids.csv")
        path_moderate = PATH_TO_SUBSET_MODERATE.joinpath("ids.csv")
        path_severe = PATH_TO_SUBSET_SEVERE.joinpath("ids.csv")
        no_ah_ids_arr.tofile(str(path_no), sep=',')
        mild_ah_ids_arr.tofile(str(path_mild), sep=',')
        moderate_ah_ids_arr.tofile(str(path_moderate), sep=',')
        severe_ah_ids_arr.tofile(str(path_severe), sep=',')

    return no_ah_ids, mild_ah_ids, moderate_ah_ids, severe_ah_ids


if __name__ == "__main__":
    PATH_TO_SUBSET_NO.mkdir(exist_ok=True)
    PATH_TO_SUBSET_MILD.mkdir(exist_ok=True)
    PATH_TO_SUBSET_MODERATE.mkdir(exist_ok=True)
    PATH_TO_SUBSET_SEVERE.mkdir(exist_ok=True)

    no_ah_ids, mild_ah_ids, moderate_ah_ids, severe_ah_ids = find_ids()
    print(f"No SA size: {len(no_ah_ids)}\n")
    print(sorted(no_ah_ids))
    print(f"Mild SA size: {len(mild_ah_ids)}\n")
    print(sorted(mild_ah_ids))
    print(f"Moderate SA size: {len(moderate_ah_ids)}\n")
    print(sorted(moderate_ah_ids))
    print(f"Severe SA size: {len(severe_ah_ids)}\n")
    print(sorted(severe_ah_ids))

