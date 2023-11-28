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

    if print_summary:
        print(f"Total number of respiratory events: {n_events}")  # 1155505
        print(f"Maximum event duration: {max_duration}")  # 6552.7s -> 1h and 49min
        print(f"Number of overlapping events: {n_overlapping_events}")  # 626499 (54%)
        # print(overlapping_sets)

    subset_stats_df = pd.DataFrame()
    subset_stats_df["n_events"] = [n_events]
    subset_stats_df["max_event_duration"] = [max_duration]
    subset_stats_df["n_overlapping_events"] = [n_overlapping_events]

    return subset_stats_df, overlapping_sets


# dataset_stats_df, overlapping_sets_list = subset_stats(ids=None, print_summary=True)
# with open(Path(__file__).parent.joinpath("stats", 'overlapping_sets.plk'), 'wb') as fp:
#     pickle.dump(overlapping_sets_list, fp)
# dataset_stats_df.to_csv(Path(__file__).parent.joinpath("dataset_stats.csv"))

# Same for subset 1
best_ids = subset_1_generator.get_best_ids()
dataset_stats_df, overlapping_sets_list = subset_stats(ids=best_ids, print_summary=True)
with open(Path(PATH_TO_SUBSET1).joinpath("stats", 'overlapping_sets.plk'), 'wb') as fp:
    pickle.dump(overlapping_sets_list, fp)
dataset_stats_df.to_csv(Path(PATH_TO_SUBSET1).joinpath("stats", "subset1_stats.csv"))

event_indices = []
event_indices_df = []
for id, sub in get_subjects_by_ids_generator(subject_ids=best_ids, progress_bar=True):
    df = sub.export_to_dataframe(print_downsampling_details=False)
    events = sub.respiratory_events

    event_indices_df.extend(list(df["event_index"]))

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
subset_1_stats_df.describe()
