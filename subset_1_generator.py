import os
import numpy as np
from tqdm import tqdm
import pickle
import yaml
import shutil

# Local imports:
from common import Subject
from object_loader import get_all_subjects, all_subjects_generator, get_subjects_by_id_range

# --- START OF CONSTANTS --- #
SUBSET_SIZE = 400  # The number of subjects that will remain after screening down the whole dataset

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = config["paths"]["local"]["subject_objects_directory"]
    PATH_TO_SUBSET1 = config["paths"]["local"]["subset_1_directory"]
else:
    PATH_TO_OBJECTS = os.path.join(os.curdir, "data", "serialized-objects")
    PATH_TO_SUBSET1 = os.path.join(os.curdir, "data", "subset-1")
# --- END OF CONSTANTS --- #

path = os.path.join(PATH_TO_SUBSET1, "ids.npy")
if os.path.isfile(path):
    best_ids_arr = np.load(path)  # array to save the best subject ids
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

        print(f"Screened dataset size: {len(score_id_dict.values())+len(extra_score_id_dict)}")
        # Rank the extra relaxed screened subjects that will supplement the strictly screened subjects:
        top_screened_scores = sorted(extra_score_id_dict, reverse=True)[0:(SUBSET_SIZE - len(score_id_dict.values()))]
        best_ids = list(score_id_dict.values())  # Add all the strictly screened subjects
        best_ids.extend([extra_score_id_dict[score] for score in top_screened_scores])  # Add the extra ranked subjects
    else:
        print(f"Screened dataset size: {len(score_id_dict.values())}")
        top_screened_scores = sorted(score_id_dict, reverse=True)[0:SUBSET_SIZE]  # List with top 400 scores
        best_ids = [score_id_dict[score] for score in top_screened_scores]  # List with top 400 ids
    best_ids_arr = np.array(best_ids)  # Equivalent array
    path = os.path.join(PATH_TO_SUBSET1, "ids")
    np.save(path, best_ids_arr)

print(f"Final subset size: {len(best_ids)}")
print(best_ids)

# for id in tqdm(best_ids):
#     shutil.copy2(os.path.join(PATH_TO_OBJECTS, str(id).zfill(4)+".bin"), "D:\\mesa")
