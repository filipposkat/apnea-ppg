import os
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

score_id_dict = {int: Subject}
# Score each subject based on events:
for id, sub in all_subjects_generator():
    central_apnea_events = sub.get_events_by_concept("central_apnea")
    obstructive_apnea_events = sub.get_events_by_concept("obstructive_apnea")
    hypopnea_events = sub.get_events_by_concept("hypopnea")
    spo2_desat_events = sub.get_events_by_concept("spo2_desat")

    # Exclude subjects who do not have any of the desired events:
    if len(central_apnea_events) != 0 and len(obstructive_apnea_events) != 0 and len(hypopnea_events) != 0 and len(
            spo2_desat_events) != 0:
        aggregate_score = len(central_apnea_events) + len(obstructive_apnea_events) + len(hypopnea_events)
        score_id_dict[aggregate_score] = id


top_400_scores = sorted(score_id_dict, reverse=True)[0:400]
best_ids = [score_id_dict[score] for score in top_400_scores]
print(best_ids)