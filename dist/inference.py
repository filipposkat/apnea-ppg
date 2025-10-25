import numpy as np
import scipy
import math
import pandas as pd

from pathlib import Path
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

# Local Imports
import common


def preprocess_signals(ppg: np.ndarray, ppg_frequency: int, spo2: np.ndarray = None, spo2_frequency: int = 1,
                       target_frequency=64,
                       detrend_ppg=False, scale_ppg=False, produce_ppg_derived_signals=False) -> dict[str, np.ndarray]:
    # Trim zeros from front:
    original_length = len(ppg)
    ppg = np.trim_zeros(ppg, trim='f')
    front_zeros = original_length - len(ppg)

    # Trim zeros from back:
    tmp_len = len(ppg)
    ppg = np.trim_zeros(ppg, trim='b')
    back_zeros = tmp_len - len(ppg)

    # Adjust the SpO2 accordingly:
    if spo2 is not None:
        assert ppg_frequency % spo2_frequency == 0
        proportion = ppg_frequency // spo2_frequency
        assert proportion == len(spo2) // original_length
        front_zeros_to_drop = front_zeros * int(proportion)
        back_zeros_to_drop = back_zeros * int(proportion)
        if back_zeros_to_drop == 0:
            spo2 = spo2[front_zeros_to_drop:]
        else:
            spo2 = spo2[front_zeros_to_drop:-back_zeros_to_drop]
        assert proportion == len(spo2) / len(ppg)

    # Resample PPG:
    proportion = target_frequency // ppg_frequency
    if proportion < 1.0:
        ppg = common.downsample_to_proportion(ppg, proportion, lpf=True)
    elif proportion > 1.0:
        ppg = common.upsample_to_proportion(ppg, proportion)
    ppg = ppg.astype("float32")  # Set type to 32 bit instead of 64 to save memory

    signals = {}

    if produce_ppg_derived_signals:
        slow_ppg = common.get_slow_ppg(ppg, fs=target_frequency, normalize=scale_ppg)
        slow_ppg = slow_ppg.astype("float32")
        signals["slow_ppg"] = slow_ppg
    if detrend_ppg:
        ppg = common.get_detrended_ppg(ppg, fs=ppg_frequency)
    if scale_ppg:
        ppg = ppg / (np.percentile(ppg, 99) + 1e-8)
    if produce_ppg_derived_signals:
        # These work better if ppg is detrended
        ppg_envelope = common.get_envelope(ppg, fs=target_frequency, smooth=True, normalize=scale_ppg)
        ppg_envelope = ppg_envelope.astype("float32")
        ppg_kte = common.get_kte(ppg, fs=target_frequency, smooth=True, normalize=scale_ppg)
        ppg_kte = ppg_kte.astype("float32")
        signals["ppg_envelope"] = ppg_envelope
        signals["ppg_kte"] = ppg_kte
    signals["ppg"] = ppg

    # Resample SpO2
    if spo2 is not None:
        median_spo2_value = np.median(spo2)
        spo2[spo2 <= 60] = median_spo2_value

        proportion = target_frequency // spo2_frequency
        if proportion < 1.0:
            spo2 = common.downsample_to_proportion(spo2, proportion, lpf=True)
        elif proportion > 1.0:
            spo2 = common.upsample_to_proportion(spo2, proportion)

        spo2 = spo2.astype("float32")
        signals["spo2"] = spo2
    return signals


def get_windows(preprocessed_ppg: np.ndarray, preprocessed_spo2=None, window_samples_size=60 * 64):
    if preprocessed_spo2 is not None:
        array = np.stack((preprocessed_spo2, preprocessed_ppg), axis=1).reshape(-1, 2)
    else:
        array = preprocessed_ppg.reshape(-1, 1)
    # Take equal-sized windows with a specified step:
    # Calculate the number of windows:
    num_windows_test = len(preprocessed_ppg) // window_samples_size

    # Generate equal-sized windows:
    windows = [preprocessed_ppg[i * window_samples_size:i * window_samples_size + window_samples_size]
               for i in range(num_windows_test)]
    # Note that when using df.iloc[] or df[], the stop part is not included. However ,when using loc stop is included

    return np.array(windows, dtype="float32")  # shape= (num of windows in test, WINDOW_SAMPLES_SIZE, numOfSignals)


def get_net_predictions(in_windows: np.ndarray, net_checkpoint_file: Path, device="cpu"):
    # Prepare tensors
    X_arr = np.swapaxes(in_windows, axis1=1, axis2=2)

    # Convert to tensors:
    X = torch.tensor(X_arr)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset=dataset, batch_size=8192, shuffle=False)

    sample_batch_input = next(iter(loader))
    window_size = sample_batch_input.shape[2]

    # Load Network:
    model_path = net_checkpoint_file
    if isinstance(device, str) and "ocl" in device:
        state = torch.load(model_path, map_location={"cpu": "cpu", "cuda:0": "cpu"}, weights_only=False)
    else:
        state = torch.load(model_path, map_location={"cpu": "cpu", "cuda:0": str(device)}, weights_only=False)

    net_class = state["net_class"]
    net_state = state["net_state_dict"]
    model_kwargs = state["net_kwargs"]

    if "lstm_max_channels" in model_kwargs:
        model_kwargs.pop("lstm_max_channels")

    net = net_class(**model_kwargs)
    net.load_state_dict(net_state)

    # Switch to eval mode:
    net.eval()
    net = net.to(device)
    probs = []
    preds = []
    with torch.no_grad():
        for (batch_i, data) in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            batch_inputs = data

            # Convert to accepted dtypes: float32, float64, int64 and maybe more but not sure
            batch_inputs = batch_inputs.type(torch.float32)
            batch_inputs = batch_inputs.to(device)

            # Predictions:
            batch_outputs = net(batch_inputs)
            batch_output_probs = F.softmax(batch_outputs, dim=1)
            _, batch_predictions = torch.max(batch_outputs, dim=1, keepdim=False)
            preds.extend(batch_predictions.ravel().tolist())
            probs.extend(batch_output_probs.swapaxes(1, 2).reshape(-1, 5).tolist())

    return np.array(preds), np.array(probs)


def get_metamodel_input_vars(predicted_probabilities: np.ndarray, descriptive_metadata: dict,
                             predictions_frequency: int = 64,
                             raw_spo2: np.ndarray = None, spo2_frequency: int = 1):
    preds: np.ndarray = np.argmax(predicted_probabilities, axis=1, keepdims=False)
    preds = np.squeeze(preds)
    n_classes = predicted_probabilities.shape[1]

    preds_proba_f_dict = None

    est_sleep_hours = np.size(preds) / (60 * 60 * predictions_frequency)
    mean_vector = predicted_probabilities.mean(axis=0, keepdims=False)
    std_vector = predicted_probabilities.std(axis=0, keepdims=False, ddof=1)  # sample std is with ddof=1
    skewness_vector = scipy.stats.skew(predicted_probabilities, axis=0, keepdims=False)
    kurtosis_vector = scipy.stats.kurtosis(predicted_probabilities, axis=0, keepdims=False)
    first_quartile_vector = np.percentile(predicted_probabilities, q=25, axis=0, keepdims=False)
    median_vector = np.percentile(predicted_probabilities, q=50, axis=0, keepdims=False)
    third_quartile_vector = np.percentile(predicted_probabilities, q=75, axis=0, keepdims=False)
    cv_vector = np.divide(std_vector, mean_vector + 0.000001)

    # Calculation of other metrics:
    n_pred_clinical_events = {e: 0 for e in (1, 2, 3)}
    pred_clinical_event_durations = {e: [] for e in (1, 2, 3)}
    i = 1
    while i < len(preds):
        event = preds[i]
        if (1 <= event <= 3) and (preds[i] != preds[i - 1]):
            d = 1
            while (i + d < len(preds)) and preds[i + d] == event:
                d += 1

            if d >= predictions_frequency * 10:
                n_pred_clinical_events[event] += 1
                pred_clinical_event_durations[event].append(d / predictions_frequency)
            i += d
        else:
            i += 1

    mean_pred_clinical_event_duration = {e: np.mean(pred_clinical_event_durations[e]
                                                    if n_pred_clinical_events[e] != 0 else 0.0)
                                         for e in n_pred_clinical_events.keys()}

    # Metadata inputs:
    # Check descriptive metadata
    mandatory_metadata = ("sex", "age", "race", "height_cm", "weight_kg", "smoker_status")
    for var in mandatory_metadata:
        assert var in descriptive_metadata
    sex = int(descriptive_metadata["sex"])
    age = int(descriptive_metadata["age"])
    race = int(descriptive_metadata["race"])
    height = float(descriptive_metadata["height_cm"])
    weight = float(descriptive_metadata["weight_kg"])
    bmi = weight / ((height / 100) ** 2)
    smoker_status = float(descriptive_metadata["smoker_status"])
    if not math.isnan(smoker_status):
        smoker_status = int(smoker_status)
    else:
        smoker_status = 4
    # 0: Never smoked
    # 1: Former smoker quit more than 1 year ago
    # 2: Former smoker quit less than 1 year ago
    # 3: Current smoker
    # 4: Do not know

    # Gather variables
    tmp_list = [sex, age, race, height, weight, bmi, smoker_status]

    # Calculate ndesat3 if spo2 available
    if raw_spo2 is not None:
        ndesat3 = common.detect_desaturations_profusion(raw_spo2,
                                                        sampling_rate=spo2_frequency,
                                                        min_drop=3,
                                                        max_plateau=60,
                                                        max_fall_rate=4,
                                                        max_drop_threshold=50,
                                                        min_drop_duration=1,
                                                        max_drop_duration=None)
        est_desat3_per_hour = ndesat3 / est_sleep_hours
        tmp_list.append(est_desat3_per_hour)

    for l in range(n_classes):
        tmp_list.append(mean_vector[l])
        tmp_list.append(std_vector[l])
        tmp_list.append(skewness_vector[l])
        tmp_list.append(kurtosis_vector[l])
        tmp_list.append(first_quartile_vector[l])
        tmp_list.append(median_vector[l])
        tmp_list.append(third_quartile_vector[l])
        tmp_list.append(cv_vector[l])

    tmp_list.extend([n_pred_clinical_events[e] / est_sleep_hours for e in n_pred_clinical_events.keys()])
    tmp_list.extend([mean_pred_clinical_event_duration[e] for e in mean_pred_clinical_event_duration.keys()])

    columns = ["sex", "age", "race", "height", "weight", "bmi", "smoker_status", "est_desat3_per_hour"]
    for l in range(5):
        columns.append(f"mean_proba_l{l}")
        columns.append(f"std_proba_l{l}")
        columns.append(f"skewness_proba_l{l}")
        columns.append(f"kurtosis_proba_l{l}")
        columns.append(f"q1_proba_l{l}")
        columns.append(f"median_proba_l{l}")
        columns.append(f"q3_proba_l{l}")
        columns.append(f"cv_proba_l{l}")

    for e in (1, 2, 3):
        columns.append(f"pred_events_per_hour_l{e}")

    for e in (1, 2, 3):
        columns.append(f"mean_pred_event_duration_l{e}")

    meta_dict = {}
    for i in range(len(columns)):
        col = columns[i]
        data = tmp_list[i]
        meta_dict[col] = [data]

    meta_df = pd.DataFrame(data=meta_dict)
    return meta_df


def ahi_to_category(ahi: float) -> str:
    if ahi < 5.0:
        # No
        return "none"
    elif ahi < 15.0:
        # Mild
        return "mild"
    elif ahi < 30.0:
        # Moderate
        return "moderate"
    else:
        # Severe
        return "severe"


def get_metamodel_predictions(input_df: pd.DataFrame, metamodels_dir: Path):
    meta_df = input_df

    if meta_df.isnull().values.any():
        print("Dataframe contains NaN values.")

        for col in meta_df.columns:
            if meta_df[col].isnull().any():
                print(f"{col}: {meta_df[col].isnull().sum()}")

        print("Replacing NaNs with median")
        meta_df.fillna(meta_df.median(axis=0), inplace=True)
        assert not meta_df.isnull().values.any()

    meta_df["norm_duration_l12"] = meta_df["norm_duration_l1"] + meta_df["norm_duration_l2"]
    meta_df["norm_duration_l123"] = meta_df["norm_duration_l1"] + meta_df["norm_duration_l2"] + meta_df[
        "norm_duration_l3"]

    if "gender" in meta_df.columns:
        metadata_columns = ["gender", "age", "race", "height", "weight", "bmi", "smoker_status"]
    else:
        metadata_columns = ["sex", "age", "race", "height", "weight", "bmi", "smoker_status"]

    if "desat3_per_hour" in meta_df.columns:
        metadata_columns.append("desat3_per_hour")

    out_columns = ["norm_duration_l0", "norm_duration_l1", "norm_duration_l2", "norm_duration_l3", "norm_duration_l4",
                   "norm_duration_l12", "norm_duration_l123",
                   "ahi_a0h3a", "ahi_c0h3a", "ahi_o0h3a",
                   "mean_apnea_clinical_event_duration", "mean_hypopnea_clinical_event_duration",
                   "mean_apneaHypopnea_clinical_event_duration",
                   "apnea_apneaHypopnea_ratio", "hypopnea_apneaHypopnea_ratio"]

    in_columns = [c for c in meta_df.columns if c not in out_columns]

    # Standardization:
    X_df: pd.DataFrame = meta_df[in_columns]
    X = X_df.to_numpy()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    predictions = {}
    for column_name in out_columns:
        if column_name in ("ahi_a0h3a", "ahi_c0h3a", "ahi_o0h3a", "mean_hypopnea_clinical_event_duration"):
            # SVM
            path = metamodels_dir / f'svr_{column_name}.pkl'
            scaler_path = metamodels_dir / f'svr_{column_name}_out_scaler.pkl'
            with open(str(path), mode="rb") as f:
                model = pickle.load(f)
            with open(str(scaler_path), mode="rb") as f:
                scaler = pickle.load(f)

            y_standardized = model.predict(X)
            y = scaler.inverse_transform(y_standardized)
        else:
            # Ridge
            path = metamodels_dir / f'ridge_{column_name}.pkl'
            with open(str(path), mode="rb") as f:
                model = pickle.load(f)
            y = model.predict(X)

        predictions[column_name] = y

    predictions["ahi_category"] = ahi_to_category(predictions["ahi_a0h3a"])

    return predictions
