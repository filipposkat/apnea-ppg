import math
from pathlib import Path
import numpy as np
import scipy
from tqdm import tqdm
import math

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import random
# from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.fft import rfft, rfftfreq
from torch.multiprocessing import Pool
# Local imports:
from common import Subject
from data_loaders_mapped import get_subject_train_test_split

# --- START OF CONSTANTS --- #
SUBSET = "0-60s"
EPOCH = 6
CREATE_DATA = True
SKIP_EXISTING_IDS = False
SIGNALS_FREQUENCY = 64  # The frequency used in the exported signals
TEST_SIZE = 0.25
SEED = 33
COMPUTE_FOURIER_TRANSFORM = False
FOURIER_COMPONENTS = 100

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
    if f"subset_0_continuous_testing_directory" in config["paths"]["local"]:
        PATH_TO_SUBSET0_CONT_TESTING = Path(
            config["paths"]["local"][f"subset_0_continuous_testing_directory"])
    else:
        PATH_TO_SUBSET0_CONT_TESTING = PATH_TO_SUBSET0
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
    PATH_TO_SUBSET0_CONT_TESTING = PATH_TO_SUBSET0
    PATH_TO_SUBSET_CONT_TESTING = PATH_TO_SUBSET
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_1_training_directory"])
    NET_TYPE = "UResIncNet"
    IDENTIFIER = "ks3-depth8-strided-0"

TRAIN_IDS, _ = get_subject_train_test_split()
# TRAIN_IDS = [27, 64, 133, 140, 183, 194, 196, 220, 303, 332, 346, 381, 405, 407, 435, 468, 490, 505, 527, 561, 571,
#              589, 628, 643, 658, 712, 713, 715, 718, 719, 725, 728, 743, 744, 796, 823, 860, 863, 892, 912, 917,
#              931, 934, 937, 939, 951, 1013, 1017, 1019, 1087, 1089, 1128, 1133, 1161, 1212, 1224, 1236, 1263, 1266,
#              1278, 1281, 1291, 1301, 1328, 1342, 1376, 1464, 1478, 1497, 1501, 1502, 1552, 1562, 1573, 1623, 1626,
#              1656, 1693, 1733, 1738, 1790, 1797, 1809, 1833, 1838, 1874, 1879, 1906, 1913, 1914, 1924, 1983, 2003,
#              2024, 2039, 2105, 2106, 2118, 2204, 2208, 2216, 2227, 2239, 2246, 2251, 2264, 2276, 2291, 2292, 2317,
#              2345, 2375, 2397, 2451, 2452, 2467, 2468, 2523, 2539, 2572, 2614, 2665, 2701, 2735, 2781, 2798, 2800,
#              2802, 2819, 2834, 2848, 2877, 2879, 2881, 2897, 2915, 2934, 2995, 3012, 3024, 3028, 3106, 3149, 3156,
#              3204, 3223, 3236, 3275, 3280, 3293, 3324, 3337, 3347, 3352, 3419, 3439, 3452, 3468, 3555, 3564, 3575,
#              3591, 3603, 3604, 3652, 3690, 3702, 3711, 3734, 3743, 3770, 3781, 3803, 3833, 3852, 3854, 3867, 3902,
#              3933, 3934, 3967, 3974, 3980, 3987, 3992, 4029, 4038, 4085, 4099, 4123, 4128, 4157, 4163, 4205, 4228,
#              4250, 4252, 4254, 4256, 4295, 4296, 4330, 4332, 4428, 4462, 4496, 4497, 4511, 4541, 4544, 4554, 4592,
#              4624, 4661, 4734, 4820, 4826, 4878, 4912, 4948, 5029, 5053, 5063, 5075, 5096, 5101, 5118, 5137, 5155,
#              5162, 5163, 5179, 5203, 5214, 5232, 5276, 5283, 5308, 5339, 5357, 5358, 5365, 5387, 5395, 5433, 5457,
#              5472, 5480, 5491, 5503, 5565, 5580, 5662, 5686, 5697, 5703, 5753, 5788, 5798, 5845, 5897, 5909, 5954,
#              5982, 6009, 6022, 6047, 6050, 6052, 6074, 6077, 6117, 6174, 6180, 6244, 6261, 6274, 6279, 6280, 6291,
#              6316, 6318, 6322, 6351, 6366, 6390, 6417, 6422, 6492, 6528, 6549, 6616, 6682, 6695, 6704, 6755, 6781,
#              6804, 6807, 6811]


# --- END OF CONSTANTS --- #

def get_metadata(sub_id: int) -> pd.Series:
    # Check if it exists in dataset-all
    if PATH_TO_SUBSET0_CONT_TESTING is not None:
        subject_arrs_path = PATH_TO_SUBSET0_CONT_TESTING.joinpath("cont-test-arrays", str(sub_id).zfill(4))
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
    if PATH_TO_SUBSET0_CONT_TESTING is not None:
        results_path = PATH_TO_SUBSET0_CONT_TESTING.joinpath("cont-test-results", str(NET_TYPE), str(IDENTIFIER),
                                                             f"epoch-{EPOCH}")
        if sub_id in TRAIN_IDS:
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
    if sub_id in TRAIN_IDS:
        results_path = results_path.joinpath("validation-subjects")
    else:
        results_path = results_path.joinpath("cross-test-subjects")
    matlab_file = results_path.joinpath(f"cont_test_signal_{sub_id}.mat")

    matlab_dict = scipy.io.loadmat(str(matlab_file))
    return matlab_dict


def get_columns_of_subject(sub_id: int) -> (int, list):
    matlab_dict = get_predictions(sub_id)
    preds_proba: np.ndarray = matlab_dict["prediction_probabilities"]
    preds: np.ndarray = matlab_dict["predictions"]
    labels: np.ndarray = matlab_dict["labels"]
    n_classes = preds_proba.shape[1]
    # def moving_average(a, n=SIGNALS_FREQUENCY*60):
    #     ret = np.cumsum(a, dtype=float)
    #     ret[n:] = ret[n:] - ret[:-n]
    #     return ret[n - 1:] / n
    #
    # preds_proba_ma = moving_average(preds_proba[:, 2].ravel())
    # label = labels == 2
    # labels_ma = moving_average(label)
    # plt.figure()
    # plt.plot(np.arange(0, preds_proba_ma.shape[0]), preds_proba_ma, label="Predicted probability MA")
    # plt.plot(np.arange(0, labels_ma.shape[0]), labels_ma, label="Ground truth label MA")
    # plt.legend()
    # plt.show()
    # exit()

    # Input stats:
    if COMPUTE_FOURIER_TRANSFORM:
        preds_proba_f_dict = {f"probas_l{i}_f": np.abs(rfft(preds_proba[:, i].ravel())) for i in
                              range(n_classes)}
        xf = rfftfreq(preds_proba.shape[0], 1 / SIGNALS_FREQUENCY)
        # plt.figure()
        # plt.plot(xf, np.abs(preds_proba_f_dict["probas_l2_f"]))
        # plt.show()
    else:
        preds_proba_f_dict = None

    est_sleep_hours = np.size(labels) / (60 * 60 * SIGNALS_FREQUENCY)
    mean_vector = preds_proba.mean(axis=0, keepdims=False)
    std_vector = preds_proba.std(axis=0, keepdims=False, ddof=1)  # sample std is with ddof=1
    skewness_vector = scipy.stats.skew(preds_proba, axis=0, keepdims=False)
    kurtosis_vector = scipy.stats.kurtosis(preds_proba, axis=0, keepdims=False)
    first_quartile_vector = np.percentile(preds_proba, q=25, axis=0, keepdims=False)
    median_vector = np.percentile(preds_proba, q=50, axis=0, keepdims=False)
    third_quartile_vector = np.percentile(preds_proba, q=75, axis=0, keepdims=False)
    cv_vector = np.divide(std_vector, mean_vector + 0.000001)

    # Calculation of other metrics:
    preds = np.squeeze(preds)
    n_pred_clinical_events = {e: 0 for e in (1, 2, 3)}
    pred_clinical_event_durations = {e: [] for e in (1, 2, 3)}
    i = 1
    while i < len(preds):
        event = preds[i]
        if (1 <= event <= 3) and (preds[i] != preds[i - 1]):
            d = 1
            while (i + d < len(preds)) and preds[i + d] == event:
                d += 1

            if d >= SIGNALS_FREQUENCY * 10:
                n_pred_clinical_events[event] += 1
                pred_clinical_event_durations[event].append(d / SIGNALS_FREQUENCY)
            i += d
        else:
            i += 1

    mean_pred_clinical_event_duration = {e: np.mean(pred_clinical_event_durations[e]
                                                    if n_pred_clinical_events[e] != 0 else 0.0)
                                         for e in n_pred_clinical_events.keys()}

    # Metadata inputs:
    metadata_df = get_metadata(sub_id)

    mesaid = int(metadata_df["mesaid"])
    sex = int(metadata_df["gender1"])
    age = int(metadata_df["sleepage5c"])
    race = int(metadata_df["race1c"])
    height = float(metadata_df["htcm5"])
    weight = float(metadata_df["wtlb5"]) * 0.45359237  # lb to kg
    bmi = float(metadata_df["nsrr_bmi"])
    smoker_status = float(metadata_df["smkstat5"])
    if not math.isnan(smoker_status):
        smoker_status = int(smoker_status)
    else:
        smoker_status = 4
    # 0: Never smoked
    # 1: Former smoker quit more than 1 year ago
    # 2: Former smoker quit less than 1 year ago
    # 3: Current smoker
    # 4: Do not know

    # Output stats:
    labels = np.squeeze(labels)
    n_clinical_events = {e: 0 for e in (1, 2, 3)}
    clinical_event_durations = {e: [] for e in (1, 2, 3)}
    i = 1
    while i < len(labels):
        event = labels[i]
        if (1 <= event <= 3) and (labels[i] != labels[i - 1]):
            d = 1
            while (i + d < len(labels)) and labels[i + d] == event:
                d += 1

            if d >= SIGNALS_FREQUENCY * 10:
                n_clinical_events[event] += 1
                clinical_event_durations[event].append(d / SIGNALS_FREQUENCY)
            i += d
        else:
            i += 1

    n_apnea_clinical_events = n_clinical_events[1] + n_clinical_events[2]
    mean_apnea_clinical_event_duration = np.mean([*clinical_event_durations[1], *clinical_event_durations[2]]) \
        if n_apnea_clinical_events != 0 else 0.0

    n_hypopnea_clinical_events = n_clinical_events[3]
    mean_hypopnea_clinical_event_duration = np.mean(clinical_event_durations[3]) \
        if n_hypopnea_clinical_events != 0 else 0.0

    ah = n_apnea_clinical_events + n_hypopnea_clinical_events
    mean_apneaHypopnea_clinical_event_duration = \
        np.mean([*clinical_event_durations[1], *clinical_event_durations[2], *clinical_event_durations[3]]) \
            if ah != 0 else 0.0

    if n_apnea_clinical_events == 0:
        apnea_apneaHypopnea_ratio = 0.0
    else:
        apnea_apneaHypopnea_ratio = n_apnea_clinical_events / ah
    if n_hypopnea_clinical_events == 0:
        hypopnea_apneaHypopnea_ratio = 0.0
    else:
        hypopnea_apneaHypopnea_ratio = n_hypopnea_clinical_events / ah

    N = labels.size
    normalized_duration_vector = np.zeros(5)
    mean_event_duration_vector = np.zeros(5)
    for lbl in range(5):
        normalized_duration_vector[lbl] = np.sum(labels == lbl) / N

    ahi_a0h3a = float(metadata_df["ahi_a0h3a"])
    ahi_c0h3a = float(metadata_df["ahi_c0h3a"])
    ahi_o0h3a = float(metadata_df["ahi_o0h3a"])
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

    tmp_list = [sex, age, race, height, weight, bmi, smoker_status]
    # tmp_list.append(est_sleep_hours)
    for l in range(n_classes):
        tmp_list.append(mean_vector[l])
        tmp_list.append(std_vector[l])
        tmp_list.append(skewness_vector[l])
        tmp_list.append(kurtosis_vector[l])
        tmp_list.append(first_quartile_vector[l])
        tmp_list.append(median_vector[l])
        tmp_list.append(third_quartile_vector[l])
        tmp_list.append(cv_vector[l])
        if COMPUTE_FOURIER_TRANSFORM:
            for f in range(FOURIER_COMPONENTS):
                tmp_list.append(preds_proba_f_dict[f"probas_l{l}_f"][f])

    tmp_list.extend([n_pred_clinical_events[e] / est_sleep_hours for e in n_pred_clinical_events.keys()])
    tmp_list.extend([mean_pred_clinical_event_duration[e] for e in mean_pred_clinical_event_duration.keys()])

    # Outputs:
    tmp_list.extend([normalized_duration_vector[l] for l in range(n_classes)])
    tmp_list.append(ahi_a0h3a)
    tmp_list.append(ahi_c0h3a)
    tmp_list.append(ahi_o0h3a)
    tmp_list.append(cat)
    tmp_list.append(mean_apnea_clinical_event_duration)
    tmp_list.append(mean_hypopnea_clinical_event_duration)
    tmp_list.append(mean_apneaHypopnea_clinical_event_duration)
    tmp_list.append(apnea_apneaHypopnea_ratio)
    tmp_list.append(hypopnea_apneaHypopnea_ratio)

    return mesaid, tmp_list


if __name__ == "__main__":
    print(f"Using the subject subset: {SUBSET}")
    print(f"The compatibility was selected to be with the subset (set the models were trained on): {subset_id}")
    print(f"Signals frequency: {SIGNALS_FREQUENCY}. If this is wrong delete dataframes and restart.")

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

    TRAIN_IDS, _ = get_subject_train_test_split()
    ids_ex_train = [id for id in ids if id not in TRAIN_IDS]
    meta_ids = ids_ex_train
    # meta_test_ids = rng.sample(meta_ids, int(TEST_SIZE * len(meta_ids)))  # does not include any original train ids
    # meta_train_ids = [id for id in meta_ids if id not in meta_test_ids]  # # does not include any original train ids

    matlab_dict = get_predictions(meta_ids[0])
    preds_proba: np.ndarray = matlab_dict["prediction_probabilities"]
    N_CLASSES = preds_proba.shape[1]
    print(f"# of Classes detected: {N_CLASSES}")

    if CREATE_DATA:
        mesaids = []
        data_list = []
        columns = ["sex", "age", "race", "height", "weight", "bmi", "smoker_status"]
        # columns.append("total_hours")
        for l in range(N_CLASSES):
            columns.append(f"mean_proba_l{l}")
            columns.append(f"std_proba_l{l}")
            columns.append(f"skewness_proba_l{l}")
            columns.append(f"kurtosis_proba_l{l}")
            columns.append(f"q1_proba_l{l}")
            columns.append(f"median_proba_l{l}")
            columns.append(f"q3_proba_l{l}")
            columns.append(f"cv_proba_l{l}")
            if COMPUTE_FOURIER_TRANSFORM:
                for f in range(FOURIER_COMPONENTS):
                    columns.append(f"l{l}_f{f}")

        for e in (1, 2, 3):
            columns.append(f"pred_events_per_hour_l{e}")

        for e in (1, 2, 3):
            columns.append(f"mean_pred_event_duration_l{e}")

        # Outputs:
        for l in range(N_CLASSES):
            columns.append(f"norm_duration_l{l}")

        columns.extend(
            ["ahi_a0h3a", "ahi_c0h3a", "ahi_o0h3a", "ahi_category",
             "mean_apnea_clinical_event_duration", "mean_hypopnea_clinical_event_duration",
             "mean_apneaHypopnea_clinical_event_duration",
             "apnea_apneaHypopnea_ratio", "hypopnea_apneaHypopnea_ratio"])

        with Pool(processes=4) as pool:
            iters = len(meta_ids)
            with tqdm(total=iters) as pbar:
                for (mesaid, tmp_list) in pool.imap_unordered(get_columns_of_subject, meta_ids):
                    mesaids.append(mesaid)
                    data_list.append(tmp_list)
                    pbar.update()

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
