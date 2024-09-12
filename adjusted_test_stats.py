from typing import Literal

import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import datetime, time
import numpy as np
import pandas as pd
import seaborn as sns
import json
from tqdm import tqdm

# Local imports:
from pre_batched_dataloader import get_pre_batched_test_loader, get_pre_batched_test_cross_sub_loader
from tester import load_confusion_matrix, accuracy_by_class, precision_by_class, \
    recall_by_class, specificity_by_class, f1_by_class, micro_average_accuracy, micro_average_precision, \
    micro_average_recall, micro_average_specificity, micro_average_f1, macro_average_accuracy, \
    macro_average_precision, \
    macro_average_recall, macro_average_specificity, macro_average_f1

if __name__ == "__main__":
    from trainer import get_last_batch

# --- START OF CONSTANTS --- #
EPOCH = 4
DESIRED_CLASSES = 4
NORMALIZE: Literal["true", "pred", "all", "none"] = "true"
CROSS_SUBJECT_TESTING = False

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])

    if "convert_spo2desat_to_normal" in config["variables"]["dataset"]:
        CONVERT_SPO2DESAT_TO_NORMAL = config["variables"]["dataset"]["convert_spo2desat_to_normal"]
    else:
        CONVERT_SPO2DESAT_TO_NORMAL = False

    if "n_input_channels" in config["variables"]["dataset"]:
        N_INPUT_CHANNELS = config["variables"]["dataset"]["n_input_channels"]
    else:
        N_INPUT_CHANNELS = 1

    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{subset_id}_directory"])
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{subset_id}_training_directory"])
    if f"subset_{subset_id}_saved_models_directory" in config["paths"]["local"]:
        MODELS_PATH = Path(config["paths"]["local"][f"subset_{subset_id}_saved_models_directory"])
    else:
        MODELS_PATH = PATH_TO_SUBSET_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = config["system"]["specs"]["compute_platform"]
    NET_TYPE = config["variables"]["models"]["net_type"]
    IDENTIFIER = config["variables"]["models"]["net_identifier"]
else:
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_TRAINING = PATH_TO_SUBSET
    CONVERT_SPO2DESAT_TO_NORMAL = False
    N_INPUT_CHANNELS = 1
    MODELS_PATH = PATH_TO_SUBSET_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = "cpu"
    NET_TYPE: str = "UResIncNet"  # UNET or UResIncNet
    IDENTIFIER: str = "ks3-depth8-strided-0"  # "ks5-depth5-layers2-strided-0" or "ks3-depth8-strided-0"


# --- END OF CONSTANTS --- #

def merge_sum_rows(array: np.ndarray, indices_dict: dict[int: list[int]]):
    """
    :param array:
    :param indices_dict: Dictionary of lists, where keys are the new row indices and values are lists with the indices
     of old rows to be merged to each new row
    :return:  The transformed array
    """
    n_new_rows = len(indices_dict.keys())
    assert n_new_rows < array.shape[0]
    assert 1 <= len(indices_dict[0]) <= array.shape[0]
    for k in range(n_new_rows):
        assert k in indices_dict.keys()

    return np.vstack([np.sum(array[indices_dict[r], :], axis=0) for r in range(n_new_rows)])


def merge_sum_columns(array: np.ndarray, indices_dict: dict[int: list[int]]):
    """
    :param array:
    :param indices_dict: Dictionary of lists, where keys are the new column indices and values are lists with the
    indices of old columns to be merged to each new column
    :return:  The transformed array
    """
    n_new_cols = len(indices_dict.keys())
    assert n_new_cols < array.shape[1]
    assert 1 <= len(indices_dict[0]) <= array.shape[1]
    for k in range(n_new_cols):
        assert k in indices_dict.keys()

    return np.vstack([np.sum(array[:, indices_dict[c]], axis=1) for c in range(n_new_cols)]).T


def merge_sum_array(array: np.ndarray, indices_dict: dict[int: list[int]]):
    """
    :param array: nxn array
    :param indices_dict: Dictionary of lists, where keys are the new row/column indices and values are lists with the
    indices of old row/columns to be merged to each new row/column
    :return:  The transformed array
    """
    assert array.shape[0] == array.shape[1]
    row_merged = merge_sum_rows(array, indices_dict=indices_dict)
    return merge_sum_columns(row_merged, indices_dict=indices_dict)


def get_stats_from_cm(cm: np.ndarray, classes: list[str]):
    total_pred = np.sum(cm)
    tps = np.diag(cm)
    fps = np.sum(cm, axis=0, keepdims=False) - tps
    fns = np.sum(cm, axis=1, keepdims=False) - tps
    tns = total_pred - (fps + fns + tps)
    correct_pred = np.sum(tps)

    tp = {}
    tn = {}
    fp = {}
    fn = {}
    for c, name in enumerate(classes):
        tp[name] = tps[c]
        tn[name] = tns[c]
        fp[name] = fps[c]
        fn[name] = fns[c]

    return total_pred, correct_pred, tp, tn, fp, fn


def get_metrics_from_cm(cm: np.array, classes: list[str], verbose=False):
    total_pred, correct_pred, tp, tn, fp, fn = get_stats_from_cm(cm, classes=classes)
    aggregate_acc = correct_pred / total_pred

    acc_by_class = accuracy_by_class(tp, tn, fp, fn, print_accuracies=verbose)
    prec_by_class = precision_by_class(tp, fp, print_precisions=verbose)
    rec_by_class = recall_by_class(tp, fn, print_recalls=verbose)
    spec_by_class = specificity_by_class(tn, fp, print_specificity=verbose)
    f1_per_class = f1_by_class(tp, fp, fn, print_f1s=verbose)

    micro_acc = micro_average_accuracy(tp, tn, fp, fn, print_accuracy=verbose)
    micro_prec = micro_average_precision(tp, fp, print_precision=verbose)
    micro_rec = micro_average_recall(tp, fn, print_recall=verbose)
    micro_spec = micro_average_specificity(tn, fp, print_specificity=verbose)
    micro_f1 = micro_average_f1(tp, fp, fn, print_f1=verbose)

    macro_acc = macro_average_accuracy(tp, tn, fp, fn, print_accuracy=verbose)
    macro_prec = macro_average_precision(tp, fp, print_precision=verbose)
    macro_rec = macro_average_recall(tp, fn, print_recall=verbose)
    macro_spec = macro_average_specificity(tn, fp, print_specificity=verbose)
    macro_f1 = macro_average_f1(tp, fp, fn, print_f1=verbose)

    metrics = {"aggregate_accuracy": aggregate_acc,
               "macro_accuracy": macro_acc,
               "macro_precision": macro_prec,
               "macro_recall": macro_rec,
               "macro_spec": macro_spec,
               "macro_f1": macro_f1,
               "micro_accuracy": micro_acc,
               "micro_precision": micro_prec,
               "micro_recall": micro_rec,
               "micro_spec": micro_spec,
               "micro_f1": micro_f1,
               "accuracy_by_class": acc_by_class,
               "precision_by_class": prec_by_class,
               "recall_by_class": rec_by_class,
               "f1_by_class": f1_per_class,
               "specificity_by_class": spec_by_class}
    return metrics


def classification_performance(cm, test=True, plot_confusion=True, target_labels=None,
                               normalize: Literal["true", "pred", "all", "none"] = "none"):
    """
    :param target_labels:
    :param plot_confusion:
    :param cm:
    :param test:
    :param normalize: {'true', 'pred', 'all'} or None
    """
    train_test = "test" if test else "train"
    metrics = get_metrics_from_cm(cm, classes=target_labels)
    accuracy = metrics["aggregate_accuracy"]
    macro_f1 = metrics["macro_f1"]

    if plot_confusion:
        if not target_labels:
            df_cm_abs = pd.DataFrame(cm, copy=True)
        else:
            df_cm_abs = pd.DataFrame(cm, index=target_labels,
                                     columns=target_labels, copy=True)

        if normalize == "true":
            for r in range(cm.shape[0]):
                s = np.sum(cm[r, :])
                s = 1 if s == 0 else s
                cm[r, :] = cm[r, :] / s
        elif normalize == "pred":
            for c in range(cm.shape[1]):
                s = np.sum(cm[:, c])
                s = 1 if s == 0 else s
                cm[:, c] = cm[:, c] / s
        elif normalize == "all":
            cm = cm / np.sum(cm)

        if not target_labels:
            df_cm = pd.DataFrame(cm, copy=True)
        else:
            df_cm = pd.DataFrame(cm, index=target_labels,
                                 columns=target_labels, copy=True)

        plt.figure(figsize=(10, 7))
        plt.title(f"Accuracy ({train_test}): {100 * accuracy:.2f}%. \n Macro F1: {100 * macro_f1:.2f}%")
        print(f"Macro F1: {100 * macro_f1:.2f}%")
        sns.set_theme(font_scale=1)  # for label size
        if normalize:
            sns.heatmap(df_cm, annot=df_cm_abs, annot_kws={'va': 'top'}, fmt=".0f", cbar=False)
            sns.heatmap(df_cm, annot=df_cm, annot_kws={'va': 'bottom'}, fmt=".2f", cbar=True)
        else:
            sns.heatmap(df_cm_abs, annot=True, fmt="d", cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    else:
        print(f"Accuracy ({train_test})  %0.1f%% " % (accuracy * 100))
        print(f"Macro F1: {100 * macro_f1:.2f}%")
    return metrics


def merged_classes_assesment(cm_arr: np.ndarray, desired_classes=DESIRED_CLASSES, normalize=NORMALIZE):
    n_classes = cm_arr.shape[0]
    metrics = None
    if n_classes == desired_classes:
        classes = ["Normal", "Central Apnea", "Obstructive Apnea", "Hypopnea", "SpO2 Desaturation"]
        classes = classes[0:n_classes]
        metrics = classification_performance(cm_arr, test=True, plot_confusion=True, target_labels=classes,
                                             normalize=NORMALIZE)
    elif desired_classes == 4:
        # Combine classes Normal SpO2 desat
        classes4 = ["Normal", "Central Apnea", "Obstructive Apnea", "Hypopnea"]
        merge_dict = {0: [0, 4],
                      1: [1],
                      2: [2],
                      3: [3]}

        cm4 = merge_sum_array(array=cm_arr, indices_dict=merge_dict)
        metrics = classification_performance(cm4, test=True, plot_confusion=True, target_labels=classes4,
                                             normalize=normalize)
    elif desired_classes == 3:
        # Combine also, classes Central apnea, Obstructive apnea
        if n_classes == 5:
            classes3 = ["Normal", "Apnea", "Hypopnea"]
            merge_dict = {0: [0, 4],
                          1: [1, 2],
                          2: [3]}
        else:
            assert n_classes == 4
            classes3 = ["Normal", "Apnea", "Hypopnea"]
            merge_dict = {0: [0],
                          1: [1, 2],
                          2: [3]}
        cm3 = merge_sum_array(array=cm_arr, indices_dict=merge_dict)
        metrics = classification_performance(cm3, test=True, plot_confusion=True, target_labels=classes3,
                                             normalize=normalize)
    elif desired_classes == 2:
        # Combine classes Central apnea, Obstructive apnea, hypopnea and combine normal with spo2 desat
        classes2 = ["Normal", "Apnea-Hypopnea"]
        if n_classes == 5:
            merge_dict = {0: [0, 4],
                          1: [1, 2, 3]}
        else:
            assert n_classes == 4
            merge_dict = {0: [0],
                          1: [1, 2, 3]}
        cm2 = merge_sum_array(array=cm_arr, indices_dict=merge_dict)
        metrics = classification_performance(cm2, test=True, plot_confusion=True, target_labels=classes2,
                                             normalize=normalize)

    return metrics


if __name__ == "__main__":
    print(IDENTIFIER)
    batch = get_last_batch(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=EPOCH)
    cm = load_confusion_matrix(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=EPOCH, batch=batch,
                               cross_subject=CROSS_SUBJECT_TESTING)
    cm = np.array(cm)
    metrics = merged_classes_assesment(cm, desired_classes=DESIRED_CLASSES, normalize=NORMALIZE)
