import pickle
from itertools import filterfalse, count
from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random
from sklearn.model_selection import train_test_split

import subset_1_generator
# Local imports:
from common import Subject
from object_loader import all_subjects_generator, get_subject_by_id, get_subjects_by_ids_generator, get_all_ids

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)
if config is not None:
    PATH_TO_SUBSET1 = config["paths"]["local"]["subset_1_directory"]
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")


def subset_stats(ids: list | None, print_summary=True):
    if ids is None:
        ids = get_all_ids()

    n_central_apnea_events = 0
    n_obstructive_apnea_events = 0
    n_hypopnea_events = 0
    n_spo2_events = 0
    n_events = 0
    n_overlapping_events = 0
    overlapping_sets = []
    max_duration = 0
    for id, sub in get_subjects_by_ids_generator(subject_ids=ids, progress_bar=True):
        events = sub.respiratory_events
        n_events += len(events)
        # Iterate every event for this subject:
        for i in range(len(events)):
            # Examine this event
            event = events[i]
            start = event["start"]
            duration = event["duration"]
            finish = start + duration
            concept = event["concept"]

            if concept == "central_apnea":
                n_central_apnea_events += 1
            elif concept == "obstructive_apnea":
                n_obstructive_apnea_events += 1
            elif concept == "hypopnea":
                n_hypopnea_events += 1
            elif concept == "spo2_desat":
                n_spo2_events += 1

            if duration > max_duration:
                max_duration = duration

            event_overlaps = False
            overlaps = [concept]
            for j in range(0, len(events)):
                e = events[j]
                s = e["start"]
                f = s + e["duration"]
                con = e["concept"]

                # Check if this event overlaps with the event under examination:
                if (start < s < finish) or (start < f < finish):
                    # Add event to the list of events that overlap with the event under examination:
                    overlaps.append(con)
                    # Check if the event under examination has not been marked already as overlapping:
                    if not event_overlaps:
                        # Mark it as overlapping:
                        event_overlaps = True
                        n_overlapping_events += 1

                elif s > finish:
                    # if true then next events are not overlapping for sure
                    break

            if len(overlaps) > 1:
                overlapping_sets.append(overlaps)

    set_occurances = {}
    for i in range(len(overlapping_sets)):
        set = sorted(overlapping_sets[i])
        set_string = ' '.join(set)
        if set_string not in set_occurances.keys():
            set_occurances[set_string] = 1
        else:
            set_occurances[set_string] += 1

    sets_sorted = sorted(set_occurances.keys(), key=lambda k: set_occurances[k], reverse=True)

    if print_summary:
        print(f"Total number of respiratory events: {n_events}")  # 1155505
        print(f"Maximum event duration: {max_duration}")  # 6552.7s -> 1h and 49min
        print(f"Number of overlapping events: {n_overlapping_events}")  # 626499 (54%)
        print(f"Most common overlap: {sets_sorted[0]}")
        # print(overlapping_sets)

    subset_stats = dict()
    subset_stats["n_events"] = n_events
    subset_stats["max_event_duration"] = max_duration
    subset_stats["n_overlapping_events"] = n_overlapping_events
    subset_stats["n_central_apnea_events"] = n_central_apnea_events
    subset_stats["n_obstructive_apnea_events"] = n_obstructive_apnea_events
    subset_stats["n_hypopnea_events"] = n_hypopnea_events
    subset_stats["n_spo2_events"] = n_spo2_events

    return subset_stats, set_occurances


# dataset_stats_dict, set_occurances_dict = subset_stats(ids=None, print_summary=True)
#
# with open(Path(__file__).parent.joinpath("stats", 'overlapping_set_occurances_dict.plk'), 'wb') as fp:
#     pickle.dump(set_occurances_dict, fp)
#
# dataset_stats_df = pd.DataFrame({k: [dataset_stats_dict[k]] for k in dataset_stats_dict.keys()})
# set_occurances_df = pd.DataFrame({k: [set_occurances_dict[k]] for k in set_occurances_dict.keys()})
# dataset_stats_df.to_csv(Path(PATH_TO_SUBSET1).joinpath("stats", "dataset_stats.csv"))
# set_occurances_df.to_csv(Path(PATH_TO_SUBSET1).joinpath("stats", "overlapping_sets_occurances.csv"))


# Same for subset 1
best_ids = subset_1_generator.get_best_ids()
# stats_dict, set_occurances_dict = subset_stats(ids=best_ids, print_summary=True)
#
# with open(Path(PATH_TO_SUBSET1).joinpath("stats", 'overlapping_sets_occurances_dict.plk'), 'wb') as fp:
#     pickle.dump(set_occurances_dict, fp)
#
# dataset_stats_df = pd.DataFrame({k: [stats_dict[k]] for k in stats_dict.keys()})
# set_occurances_df = pd.DataFrame({k: [set_occurances_dict[k]] for k in set_occurances_dict.keys()})
# dataset_stats_df.to_csv(Path(PATH_TO_SUBSET1).joinpath("stats", "subset1_stats.csv"))
# set_occurances_df.to_csv(Path(PATH_TO_SUBSET1).joinpath("stats", "overlapping_sets_occurances.csv"))

event_indices = []
event_indices_df = []
for id, sub in get_subjects_by_ids_generator(subject_ids=best_ids, progress_bar=True):
    df = sub.export_to_dataframe(print_downsampling_details=False)
    events = sub.respiratory_events

    event_indices_df.extend(df["event_index"].tolist())

    for event in events:
        concept = event["concept"]
        if event["concept"] == "central_apnea":
            event_indices.append(1)
        elif event["concept"] == "obstructive_apnea":
            event_indices.append(2)
        elif event["concept"] == "hypopnea":
            event_indices.append(3)
        elif event["concept"] == "spo2_desat":
            event_indices.append(4)

subset_1_stats_df = pd.DataFrame()
subset_1_stats_df["Events"] = pd.Series(event_indices)
subset_1_stats_df["Events_after_downsampling"] = pd.Series(event_indices_df)
subset_1_stats_df.to_csv(Path(PATH_TO_SUBSET1).joinpath("stats", "subset_1_event_stats.csv"))
print(subset_1_stats_df.describe())
