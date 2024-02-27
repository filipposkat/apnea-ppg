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
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
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
    min_duration = 9999999

    max_ca_duration = 0
    min_ca_duration = 999999
    max_oa_duration = 0
    min_oa_duration = 999999
    max_h_duration = 0
    min_h_duration = 999999
    max_spo2_duration = 0
    min_spo2_duration = 999999
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
                if duration > max_ca_duration:
                    max_ca_duration = duration
                elif duration < min_ca_duration:
                    min_ca_duration = duration
            elif concept == "obstructive_apnea":
                n_obstructive_apnea_events += 1
                if duration > max_oa_duration:
                    max_oa_duration = duration
                elif duration < min_oa_duration:
                    min_oa_duration = duration
            elif concept == "hypopnea":
                n_hypopnea_events += 1
                if duration > max_h_duration:
                    max_h_duration = duration
                elif duration < min_h_duration:
                    min_h_duration = duration
            elif concept == "spo2_desat":
                n_spo2_events += 1
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
    subset_stats["max_event_duration"] = max_duration
    subset_stats["min_event_duration"] = min_duration
    subset_stats["max_central_apnea_duration"] = max_ca_duration
    subset_stats["min_central_apnea_duration"] = min_ca_duration
    subset_stats["max_obstructive_apnea_duration"] = max_oa_duration
    subset_stats["min_obstructive_apnea_duration"] = min_oa_duration
    subset_stats["max_hyponea_duration"] = max_h_duration
    subset_stats["min_hyponea_duration"] = min_h_duration
    subset_stats["max_spo2_desat_duration"] = max_spo2_duration
    subset_stats["min_spo2_desat_duration"] = min_spo2_duration
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
# best_ids = subset_1_generator.get_best_ids()
best_ids = [107, 1212, 490, 4252, 5232, 5179, 1342, 5451, 4948, 332, 155, 863, 937, 675, 6616, 5788, 6291, 5053, 718, 2771, 386, 1301, 5358, 2039, 744, 133, 5845, 6280, 6755, 3042, 125, 3280, 1797, 3452, 2421, 3604, 5714, 4661, 1983, 3352, 1128, 1464, 3037, 5182, 1879, 2394, 5986, 6704, 1266, 5096, 5686, 1161, 2467, 3933, 561, 5395, 626, 4229, 5203, 5480, 5503, 628, 931, 3419, 1790, 5396, 1650, 2216, 4624, 2848, 4250, 1497, 4826, 6318, 5697, 1838, 571, 3439, 3603, 3564, 743, 4541, 6695, 4085, 3293, 6193, 2251, 2651, 4157, 5283, 5695, 2239, 5276, 468, 935, 1236, 5357, 572, 2539, 3987, 2126, 6322, 712, 811, 4014, 4428, 715, 2636, 5155, 2819, 2276, 4038, 5163, 5608, 2292, 1087, 2106, 4256, 2614, 2147, 3275, 4254, 5118, 1656, 6077, 2408, 1863, 6274, 3149, 3024, 6485, 1626, 3013, 505, 5308, 5801, 917, 3337, 5897, 220, 648, 1906, 2747, 5101, 6804, 4228, 4794, 1478, 2735, 2961, 796, 2118, 4511, 2834, 3012, 1874, 2781, 3025, 3591, 4554, 1356, 6261, 2105, 2881, 4734, 5162, 5909, 5954, 2434, 5214, 6417, 1756, 6022, 5847, 1089, 3468, 4088, 6180, 2291, 3992, 4295, 5982, 1924, 346, 3854, 5137, 1013, 1263, 823, 6811, 2204, 4341, 4270, 4322, 6682, 1281, 1623, 4057, 4174, 6351, 3043, 2572, 3555, 4205, 620, 713, 728, 2024, 405, 33, 1016, 1501, 5148, 5582, 5907, 2750, 2934, 2374, 4330, 5002, 6117, 4515, 3324, 5472, 6492, 1010, 2246, 4544, 5957, 1913, 2798, 1271, 1552, 5703, 589, 1573, 4497, 3770, 4592, 6244, 2879, 2915, 1589, 2208, 2264, 5231, 5580, 725, 1914, 2780, 3743, 4296, 6390, 679, 2995, 3711, 3833, 6549, 5662, 1224, 6009, 194, 4878, 2030, 3314, 5433, 2688, 6047, 6756, 3106, 4912, 651, 1738, 3823, 6807, 979, 1693, 6075, 435, 2468, 2877, 3781, 3913, 2800, 3652, 6726, 2345, 3556, 6037, 381, 2701, 6538, 3934, 5457, 196, 643, 3239, 5063, 5393, 934, 4123, 6366, 1562, 2003, 2897, 4671, 5029, 4820, 5339, 6174, 27, 3236, 3332, 1604, 3980, 860, 2397, 4029, 140, 658, 951, 1291, 6052, 1453, 1502, 3867, 4508, 2523, 6316, 1278, 3347, 3876, 719, 1733, 3469, 4478, 5387, 5491, 6279, 2452, 3028, 3852, 912, 3974, 5753, 1328, 2665, 2685, 2802, 4437, 4496, 6074, 6422, 3702, 6781, 1133, 2227, 3734, 3902, 5075, 5365, 183, 303, 407, 1376, 1809, 3967, 4462, 6697, 3294, 3803, 2317, 527, 892, 1833, 3156, 5261, 5565, 6528, 2375, 3204, 4099, 939, 3223, 5798, 6050, 2451, 4163, 1017, 6215, 6424, 1019, 3671, 3690, 4047, 4128, 5311, 64, 3575, 4332]
stats_dict, set_occurances_dict = subset_stats(ids=best_ids, print_summary=True)

PATH_TO_SUBSET1.joinpath("stats").mkdir(exist_ok=True)

with open(PATH_TO_SUBSET1.joinpath("stats", 'overlapping_sets_occurances_dict.plk'), 'wb') as fp:
    pickle.dump(set_occurances_dict, fp)

dataset_stats_df = pd.DataFrame({k: [stats_dict[k]] for k in stats_dict.keys()})
set_occurances_df = pd.DataFrame({k: [set_occurances_dict[k]] for k in set_occurances_dict.keys()})
dataset_stats_df.to_csv(PATH_TO_SUBSET1.joinpath("stats", "subset1_stats.csv"))
set_occurances_df.to_csv(PATH_TO_SUBSET1.joinpath("stats", "overlapping_sets_occurances.csv"))

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
subset_1_stats_df.to_csv(PATH_TO_SUBSET1.joinpath("stats", "subset_1_event_stats.csv"))
print(subset_1_stats_df.describe())
