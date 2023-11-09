import os
import numpy as np
from tqdm import tqdm
import pickle
import yaml

# Local imports:
from common import Subject
from object_loader import get_all_subjects, all_subjects_generator, get_subjects_by_id

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = config["paths"]["local"]["subject_objects_directory"]
    PATH_TO_SUBSET1 = config["paths"]["local"]["subset_1_directory"]
else:
    PATH_TO_OBJECTS = os.path.join(os.curdir, "data", "serialized-objects")
    PATH_TO_SUBSET1 = os.path.join(os.curdir, "data", "subset-1")

path = os.path.join(PATH_TO_SUBSET1, "ids.npy")
if os.path.isfile(path):
    best_ids_arr = np.load(path)
    best_ids = best_ids_arr.tolist()
else:
    score_id_dict = {int: Subject}
    # Score each subject based on events:
    for id, sub in all_subjects_generator():
        n_central_apnea_events = len(sub.get_events_by_concept("central_apnea"))
        n_obstructive_apnea_events = len(sub.get_events_by_concept("obstructive_apnea"))
        n_hypopnea_events = len(sub.get_events_by_concept("hypopnea"))
        n_spo2_desat_events = len(sub.get_events_by_concept("spo2_desat"))

        # Exclude subjects who do not have any of the desired events:
        if n_central_apnea_events != 0 and n_obstructive_apnea_events != 0 and n_hypopnea_events != 0 and n_spo2_desat_events != 0:
            aggregate_score = n_central_apnea_events + n_obstructive_apnea_events + n_hypopnea_events
            score_id_dict[aggregate_score] = id


    top_400_scores = sorted(score_id_dict, reverse=True)[0:400]
    best_ids = [score_id_dict[score] for score in top_400_scores]
    best_ids_arr = np.array(best_ids)
    path = os.path.join(PATH_TO_SUBSET1, "ids")
    np.save(path, best_ids_arr)

print(best_ids)
