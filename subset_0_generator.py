# Whole dataset with all subjects

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
from object_loader import get_all_ids, all_subjects_generator

# --- START OF CONSTANTS --- #
SUBSET_SIZE = 514  # The number of subjects that will remain after screening down the whole dataset
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
    PATH_TO_SUBSET = Path(config["paths"]["local"]["subset_0_directory"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "dataset-all")


# --- END OF CONSTANTS --- #
def get_ids():
    path = PATH_TO_SUBSET.joinpath("ids.npy")
    if path.is_file():
        ids_arr = np.load(str(path))  # array to save the best subject ids
        ids = ids_arr.tolist()  # equivalent list
    else:
        ids = list(get_all_ids())
        ids_arr = np.array(ids)  # Equivalent array
        path = PATH_TO_SUBSET.joinpath("ids")
        np.save(str(path), ids_arr)

    return ids.copy()


if __name__ == "__main__":
    PATH_TO_SUBSET.mkdir(exist_ok=True)
    ids = get_ids()
    print(f"Dataset size: {len(ids)}\n")
    print(ids)

