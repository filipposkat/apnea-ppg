import math
from pathlib import Path
import numpy as np
import scipy
from tqdm import tqdm

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import random
# from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Local imports:
from common import Subject

# --- START OF CONSTANTS --- #
SUBSET = 0
EPOCH = 32
CREATE_DATA = False
SKIP_EXISTING_IDS = False
WINDOW_SEC_SIZE = 16
SIGNALS_FREQUENCY = 32  # The frequency used in the exported signals
TEST_SIZE = 0.25
SEED = 33

WINDOW_SAMPLES_SIZE = WINDOW_SEC_SIZE * SIGNALS_FREQUENCY

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])
    if "subset_0_directory" in config["paths"]["local"]:
        PATH_TO_SUBSET0 = Path(config["paths"]["local"]["subset_0_directory"])
    else:
        PATH_TO_SUBSET0 = None
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{SUBSET}_directory"])
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{subset_id}_training_directory"])
    if f"subset_{SUBSET}_continuous_testing_directory" in config["paths"]["local"]:
        PATH_TO_SUBSET_CONT_TESTING = Path(
            config["paths"]["local"][f"subset_{SUBSET}_continuous_testing_directory"])
    else:
        PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    NET_TYPE = config["variables"]["models"]["net_type"]
    IDENTIFIER = config["variables"]["models"]["net_identifier"]
else:
    subset_id = 1
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET0 = None
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_1_training_directory"])
    NET_TYPE = "UResIncNet"
    IDENTIFIER = "ks3-depth8-strided-0"


# --- END OF CONSTANTS --- #

def get_metadata(sub_id: int):
    # Check if it exists in dataset-all
    if PATH_TO_SUBSET0 is not None:
        subject_arrs_path = PATH_TO_SUBSET0.joinpath("cont-test-arrays", str(sub_id).zfill(4))
        metadata_path = subject_arrs_path.joinpath("sub_metadata.csv")
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path, header=None, index_col=0).squeeze()
            return metadata_df

    # Search in subset
    subject_arrs_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-arrays", str(sub_id).zfill(4))
    metadata_path = subject_arrs_path.joinpath("sub_metadata.csv")
    metadata_df = pd.read_csv(metadata_path, header=None, index_col=0).squeeze()
    return metadata_df


def get_predictions(sub_id: int) -> dict:
    # Check if it exists in dataset-all
    if PATH_TO_SUBSET0 is not None:
        results_path = PATH_TO_SUBSET0.joinpath("cont-test-results", str(NET_TYPE), str(IDENTIFIER),
                                                f"epoch-{EPOCH}")
        if sub_id in train_ids:
            results_path = results_path.joinpath("validation-subjects")
        else:
            results_path = results_path.joinpath("cross-test-subjects")
        matlab_file = results_path.joinpath(f"cont_test_signal_{sub_id}.mat")
        if matlab_file.exists():
            matlab_dict = scipy.io.loadmat(str(matlab_file))
            return matlab_dict

    # Search in subset
    results_path = PATH_TO_SUBSET_CONT_TESTING.joinpath("cont-test-results", str(NET_TYPE), str(IDENTIFIER),
                                                        f"epoch-{EPOCH}")
    if sub_id in train_ids:
        results_path = results_path.joinpath("validation-subjects")
    else:
        results_path = results_path.joinpath("cross-test-subjects")
    matlab_file = results_path.joinpath(f"cont_test_signal_{sub_id}.mat")

    matlab_dict = scipy.io.loadmat(str(matlab_file))
    return matlab_dict


if __name__ == "__main__":
    PATH_TO_META_MODEL = PATH_TO_SUBSET_CONT_TESTING.joinpath("meta-model", f"trainedOn-subset-{subset_id}",
                                                              str(NET_TYPE), str(IDENTIFIER), f"epoch-{EPOCH}")
    PATH_TO_META_MODEL.mkdir(exist_ok=True, parents=True)

    path = PATH_TO_SUBSET.joinpath("ids.npy")
    rng = random.Random(SEED)
    if path.is_file():
        ids_arr = np.load(str(path))  # array to save the best subject ids
        ids: list = ids_arr.tolist()  # equivalent list
    else:
        print(f"Subset-{SUBSET} has no ids generated yet")
        exit(1)

    train_ids = [27, 64, 133, 140, 183, 194, 196, 220, 303, 332, 346, 381, 405, 407, 435, 468, 490, 505, 527, 561, 571,
                 589, 628, 643, 658, 712, 713, 715, 718, 719, 725, 728, 743, 744, 796, 823, 860, 863, 892, 912, 917,
                 931, 934, 937, 939, 951, 1013, 1017, 1019, 1087, 1089, 1128, 1133, 1161, 1212, 1224, 1236, 1263, 1266,
                 1278, 1281, 1291, 1301, 1328, 1342, 1376, 1464, 1478, 1497, 1501, 1502, 1552, 1562, 1573, 1623, 1626,
                 1656, 1693, 1733, 1738, 1790, 1797, 1809, 1833, 1838, 1874, 1879, 1906, 1913, 1914, 1924, 1983, 2003,
                 2024, 2039, 2105, 2106, 2118, 2204, 2208, 2216, 2227, 2239, 2246, 2251, 2264, 2276, 2291, 2292, 2317,
                 2345, 2375, 2397, 2451, 2452, 2467, 2468, 2523, 2539, 2572, 2614, 2665, 2701, 2735, 2781, 2798, 2800,
                 2802, 2819, 2834, 2848, 2877, 2879, 2881, 2897, 2915, 2934, 2995, 3012, 3024, 3028, 3106, 3149, 3156,
                 3204, 3223, 3236, 3275, 3280, 3293, 3324, 3337, 3347, 3352, 3419, 3439, 3452, 3468, 3555, 3564, 3575,
                 3591, 3603, 3604, 3652, 3690, 3702, 3711, 3734, 3743, 3770, 3781, 3803, 3833, 3852, 3854, 3867, 3902,
                 3933, 3934, 3967, 3974, 3980, 3987, 3992, 4029, 4038, 4085, 4099, 4123, 4128, 4157, 4163, 4205, 4228,
                 4250, 4252, 4254, 4256, 4295, 4296, 4330, 4332, 4428, 4462, 4496, 4497, 4511, 4541, 4544, 4554, 4592,
                 4624, 4661, 4734, 4820, 4826, 4878, 4912, 4948, 5029, 5053, 5063, 5075, 5096, 5101, 5118, 5137, 5155,
                 5162, 5163, 5179, 5203, 5214, 5232, 5276, 5283, 5308, 5339, 5357, 5358, 5365, 5387, 5395, 5433, 5457,
                 5472, 5480, 5491, 5503, 5565, 5580, 5662, 5686, 5697, 5703, 5753, 5788, 5798, 5845, 5897, 5909, 5954,
                 5982, 6009, 6022, 6047, 6050, 6052, 6074, 6077, 6117, 6174, 6180, 6244, 6261, 6274, 6279, 6280, 6291,
                 6316, 6318, 6322, 6351, 6366, 6390, 6417, 6422, 6492, 6528, 6549, 6616, 6682, 6695, 6704, 6755, 6781,
                 6804, 6807, 6811]

    ids_ex_train = [id for id in ids if id not in train_ids]
    meta_ids = ids_ex_train
    # meta_test_ids = rng.sample(meta_ids, int(TEST_SIZE * len(meta_ids)))  # does not include any original train ids
    # meta_train_ids = [id for id in meta_ids if id not in meta_test_ids]  # # does not include any original train ids

    if CREATE_DATA:
        mesaids = []
        data_list = []
        columns = ["gender", "age", "race"]
        for l in range(5):
            columns.append(f"mean_proba_l{l}")
            columns.append(f"std_proba_l{l}")
            columns.append(f"skewness_proba_l{l}")
            columns.append(f"kurtosis_proba_l{l}")
            columns.append(f"q1_proba_l{l}")
            columns.append(f"median_proba_l{l}")
            columns.append(f"q3_proba_l{l}")
            columns.append(f"cv_proba_l{l}")

        columns.extend(
            ["norm_duration_l0", "norm_duration_l1", "norm_duration_l2", "norm_duration_l3", "norm_duration_l4",
             "ahi_a0h3a", "ahi_category"])
        for sub_id in tqdm(meta_ids):
            # print(sub_id)

            matlab_dict = get_predictions(sub_id)
            preds_proba: np.ndarray = matlab_dict["prediction_probabilities"]
            preds: np.ndarray = matlab_dict["predictions"]
            labels: np.ndarray = matlab_dict["labels"]

            # Input stats:
            mean_vector = preds_proba.mean(axis=0, keepdims=False)
            std_vector = preds_proba.std(axis=0, keepdims=False)
            skewness_vector = scipy.stats.skew(preds_proba, axis=0, keepdims=False)
            kurtosis_vector = scipy.stats.kurtosis(preds_proba, axis=0, keepdims=False)
            first_quartile_vector = np.percentile(preds_proba, q=25, axis=0, keepdims=False)
            median_vector = np.percentile(preds_proba, q=50, axis=0, keepdims=False)
            third_quartile_vector = np.percentile(preds_proba, q=75, axis=0, keepdims=False)
            cv_vector = np.divide(std_vector, mean_vector + 0.000001)

            metadata_df = get_metadata(sub_id)

            mesaid = int(metadata_df["mesaid"])
            gender = int(metadata_df["gender1"])
            age = int(metadata_df["sleepage5c"])
            race = int(metadata_df["race1c"])

            # Output stats:
            N = labels.size
            normalized_duration_vector = np.zeros(5)
            for lbl in range(5):
                normalized_duration_vector[lbl] = np.sum(labels == lbl) / N
            ahi_a0h3a = float(metadata_df["ahi_a0h3a"])
            if ahi_a0h3a < 5:
                # No
                cat = 0
            elif ahi_a0h3a < 15:
                # Mild
                cat = 1
            elif ahi_a0h3a < 30:
                # Moderate
                cat = 2
            else:
                # Severe
                cat = 3

            tmp_list = [gender, age, race]
            for l in range(5):
                tmp_list.append(mean_vector[l])
                tmp_list.append(std_vector[l])
                tmp_list.append(skewness_vector[l])
                tmp_list.append(kurtosis_vector[l])
                tmp_list.append(first_quartile_vector[l])
                tmp_list.append(median_vector[l])
                tmp_list.append(third_quartile_vector[l])
                tmp_list.append(cv_vector[l])

            tmp_list.extend([normalized_duration_vector[l] for l in range(5)])
            tmp_list.append(ahi_a0h3a)
            tmp_list.append(cat)

            mesaids.append(mesaid)
            data_list.append(tmp_list)

        meta_df = pd.DataFrame(data=data_list, columns=columns, index=mesaids)
        meta_df.to_csv(PATH_TO_META_MODEL.joinpath("meta_df.csv"))
        # meta_train_df = pd.DataFrame(data=train_data_list, columns=columns)
        # meta_train_df.to_csv(PATH_TO_META_MODEL.joinpath("meta_train_df.csv"))
        # meta_test_df = pd.DataFrame(data=test_data_list, columns=columns)
        # meta_test_df.to_csv(PATH_TO_META_MODEL.joinpath("meta_test_df.csv"))
    else:
        meta_df = pd.read_csv(PATH_TO_META_MODEL.joinpath("meta_df.csv"), index_col=0)
        # meta_train_df = pd.read_csv(PATH_TO_META_MODEL.joinpath("meta_train_df.csv"), index_col=0)
        # meta_test_df = pd.read_csv(PATH_TO_META_MODEL.joinpath("meta_test_df.csv"), index_col=0)
        meta_train_df, meta_test_df = train_test_split(meta_df, test_size=TEST_SIZE, random_state=33,
                                                       shuffle=True, stratify=meta_df["ahi_category"])
        print(meta_train_df.describe())
        print(meta_test_df.describe())