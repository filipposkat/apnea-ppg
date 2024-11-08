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

# Local imports:
from common import Subject
from object_loader import all_subjects_generator, get_subject_by_id, get_subjects_by_ids_generator, get_all_ids

SUBSET_ID = 0  # Either int or "config"
FREQUENCY = 256
with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)
if config is not None:
    if SUBSET_ID == "config":
        subset_id = int(config["variables"]["dataset"]["subset"])
    else:
        subset_id = SUBSET_ID
    PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{subset_id}_directory"])
else:
    subset_id = SUBSET_ID
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")


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
    min_duration = 9999999
    ca_durations = []
    oa_durations = []
    h_durations = []
    spo2_durations = []
    max_ca_duration = 0
    min_ca_duration = 999999
    max_oa_duration = 0
    min_oa_duration = 999999
    max_h_duration = 0
    min_h_duration = 999999
    max_spo2_duration = 0
    min_spo2_duration = 999999
    total_normal_duration_estimate = 0
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
                ca_durations.append(duration)
                if duration > max_ca_duration:
                    max_ca_duration = duration
                elif duration < min_ca_duration:
                    min_ca_duration = duration
            elif concept == "obstructive_apnea":
                n_obstructive_apnea_events += 1
                oa_durations.append(duration)
                if duration > max_oa_duration:
                    max_oa_duration = duration
                elif duration < min_oa_duration:
                    min_oa_duration = duration
            elif concept == "hypopnea":
                n_hypopnea_events += 1
                h_durations.append(duration)
                if duration > max_h_duration:
                    max_h_duration = duration
                elif duration < min_h_duration:
                    min_h_duration = duration
            elif concept == "spo2_desat":
                n_spo2_events += 1
                spo2_durations.append(duration)
                if duration > max_spo2_duration:
                    max_spo2_duration = duration
                elif duration < min_spo2_duration:
                    min_spo2_duration = duration

            if duration > max_duration:
                max_duration = duration
            elif duration < min_duration:
                min_duration = duration

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

        df = sub.export_to_dataframe(print_downsampling_details=False)
        total_normal_duration_estimate += sum(df["event_index"] == 0) * FREQUENCY

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
        print(f"Minimum event duration: {min_duration}")
        print(f"Number of overlapping events: {n_overlapping_events}")  # 626499 (54%)
        print(f"Most common overlap: {sets_sorted[0]}")
        # print(overlapping_sets)

    subset_stats = dict()
    subset_stats["n_events"] = n_events

    subset_stats["min_event_duration"] = min_duration
    subset_stats["max_event_duration"] = max_duration

    subset_stats["total_normal_sleep_duration_estimate"] = total_normal_duration_estimate

    subset_stats["min_central_apnea_duration"] = min_ca_duration
    subset_stats["avg_central_apnea_duration"] = sum(ca_durations) / len(ca_durations)
    subset_stats["max_central_apnea_duration"] = max_ca_duration
    subset_stats["total_central_apnea_duration"] = sum(ca_durations)

    subset_stats["min_obstructive_apnea_duration"] = min_oa_duration
    subset_stats["avg_obstructive_apnea_duration"] = sum(oa_durations) / len(oa_durations)
    subset_stats["max_obstructive_apnea_duration"] = max_oa_duration
    subset_stats["total_obstructive_apnea_duration"] = max_oa_duration

    subset_stats["min_hyponea_duration"] = min_h_duration
    subset_stats["avg_hyponea_duration"] = sum(h_durations) / len(h_durations)
    subset_stats["max_hyponea_duration"] = max_h_duration
    subset_stats["total_hyponea_duration"] = sum(h_durations)

    subset_stats["min_spo2_desat_duration"] = min_spo2_duration
    subset_stats["avg_spo2_desat_duration"] = sum(spo2_durations) / len(spo2_durations)
    subset_stats["max_spo2_desat_duration"] = max_spo2_duration
    subset_stats["total_spo2_desat_duration"] = sum(spo2_durations)

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


path = PATH_TO_SUBSET.joinpath("ids.npy")
if path.is_file():
    best_ids_arr = np.load(str(path))  # array to save the best subject ids
    best_ids: list[int] = best_ids_arr.tolist()  # equivalent list
else:
    print(f"Subset-{subset_id} has no ids generated yet")
    exit(1)

# stats_dict, set_occurances_dict = subset_stats(ids=best_ids, print_summary=True)

# PATH_TO_SUBSET.joinpath("stats").mkdir(exist_ok=True)
# with open(PATH_TO_SUBSET.joinpath("stats", 'overlapping_sets_occurances_dict.plk'), 'wb') as fp:
#     pickle.dump(set_occurances_dict, fp)
#
# dataset_stats_df = pd.DataFrame({k: [stats_dict[k]] for k in stats_dict.keys()})
# set_occurances_df = pd.DataFrame({k: [set_occurances_dict[k]] for k in set_occurances_dict.keys()})
# dataset_stats_df.to_csv(PATH_TO_SUBSET.joinpath("stats", "subset1_stats.csv"))
# set_occurances_df.to_csv(PATH_TO_SUBSET.joinpath("stats", "overlapping_sets_occurances.csv"))


normal_events = 0
ca_events = 0
oa_events = 0
h_events = 0
spo2_events = 0
for id, sub in get_subjects_by_ids_generator(subject_ids=best_ids, progress_bar=True):
    df = sub.export_to_dataframe(print_downsampling_details=False)

    normal_events += sum(df["event_index"] == 0)
    ca_events += sum(df["event_index"] == 1)
    oa_events += sum(df["event_index"] == 2)
    h_events += sum(df["event_index"] == 3)
    spo2_events += sum(df["event_index"] == 4)

total_events = normal_events + ca_events + oa_events + h_events + spo2_events

subset_series_stats = pd.Series()
subset_series_stats["normal_events"] = normal_events
subset_series_stats["central_apnea_events"] = ca_events
subset_series_stats["obstructive_apnea_events"] = oa_events
subset_series_stats["hypopnea_events"] = h_events
subset_series_stats["spo2_events"] = spo2_events

subset_series_stats["total_events"] = total_events
subset_series_stats["normal_events_ratio"] = normal_events / total_events
subset_series_stats["central_apnea_events_ratio"] = ca_events / total_events
subset_series_stats["obstructive_apnea_events_ratio"] = oa_events / total_events
subset_series_stats["hypopnea_events_ratio"] = h_events / total_events
subset_series_stats["spo2_events_ratio"] = spo2_events / total_events

subset_series_stats.to_csv(PATH_TO_SUBSET.joinpath("stats", "subset_1_event_series_stats.csv"))

