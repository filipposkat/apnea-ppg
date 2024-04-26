import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import datetime, time
import numpy as np
import json
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import ROC, AUROC
from sklearn.metrics import auc
from torchmetrics.functional import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import Pool, set_start_method

import data_loaders_mapped

# Local imports:
from data_loaders_mapped import MappedDataset, BatchSampler
from pre_batched_dataloader import get_pre_batched_test_loader, get_pre_batched_test_cross_sub_loader
from UNet import UNet

if __name__ == "__main__":
    from trainer import get_saved_epochs, get_saved_batches, get_last_batch, get_last_epoch, load_checkpoint

# --- START OF CONSTANTS --- #
EPOCHS = 100
BATCH_SIZE_TEST = 1024
MAX_BATCHES = None  # Maximum number of test batches to use or None to use all of them
LOAD_FROM_BATCH = 0
NUM_WORKERS = 2
NUM_PROCESSES_FOR_METRICS = 6
PRE_FETCH = 2
TEST_MODEL = False
OVERWRITE_METRICS = False

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])
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
    MODELS_PATH = PATH_TO_SUBSET_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = "cpu"
    NET_TYPE: str = "UResIncNet"  # UNET or UResIncNet
    IDENTIFIER: str = "ks3-depth8-strided-0"  # "ks5-depth5-layers2-strided-0" or "ks3-depth8-strided-0"

MODELS_PATH.mkdir(parents=True, exist_ok=True)


# --- END OF CONSTANTS --- #
def save_rocs(roc_info_by_class: dict[str: dict[str: list | float]],
              net_type: str, identifier: str, epoch: int, batch: int, cross_subject=False, save_plot=True):
    if cross_subject:
        roc_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}",
                                        f"batch-{batch}-cross_test_roc.json")
    else:
        roc_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_roc.json")
    with open(roc_path, 'w') as file:
        json.dump(roc_info_by_class, file)

    if save_plot:
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.ravel()

        plot_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_roc.png")
        for c, class_name in enumerate(roc_info_by_class.keys()):
            average_fpr = roc_info_by_class[class_name]["average_fpr"]
            average_tpr = roc_info_by_class[class_name]["average_tpr"]
            average_auc = roc_info_by_class[class_name]["average_auc"]

            ax = axs[c]
            ax.plot(average_fpr, average_tpr)
            ax.set_title(f"Average ROC for class: {class_name} with average AUC: {average_auc:.2f}")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
        fig.savefig(str(plot_path))
        plt.close(fig)


def save_confusion_matrix(confusion_matrix: list[list[float]], net_type: str, identifier: str, epoch: int, batch: int,
                          cross_subject=False):
    if cross_subject:
        cm_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-cross_test_cm.json")
    else:
        cm_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_cm.json")
    with open(cm_path, 'w') as file:
        json.dump(confusion_matrix, file)


def save_metrics(metrics: dict[str: float], net_type: str, identifier: str, epoch: int, batch: int,
                 cross_subject=False):
    if cross_subject:
        metrics_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}",
                                            f"batch-{batch}-cross_test_metrics.json")
    else:
        metrics_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}",
                                            f"batch-{batch}-test_metrics.json")
    with open(metrics_path, 'w') as file:
        json.dump(metrics, file)


def load_rocs(net_type: str, identifier: str, epoch: int, batch: int, cross_subject=False) \
        -> dict[str: dict[str: list | float]] | None:
    if cross_subject:
        roc_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}",
                                        f"batch-{batch}-cross_test_roc.json")
    else:
        roc_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_roc.json")
    if roc_path.exists():
        with open(roc_path, 'r') as file:
            return json.load(file)
    else:
        return None


def load_confusion_matrix(net_type: str, identifier: str, epoch: int, batch: int, cross_subject=False) \
        -> list[list[float]] | None:
    if cross_subject:
        cm_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-cross_test_cm.json")
    else:
        cm_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_cm.json")
    if cm_path.exists():
        with open(cm_path, 'r') as file:
            return json.load(file)
    else:
        return None


def load_metrics(net_type: str, identifier: str, epoch: int, batch: int, cross_subject=False) -> dict[str: float]:
    if cross_subject:
        metrics_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}",
                                            f"batch-{batch}-cross_test_metrics.json")
    else:
        metrics_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}",
                                            f"batch-{batch}-test_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as file:
            return json.load(file)
    else:
        return None


def load_metrics_by_epoch(net_type: str, identifier: str) -> tuple[list[float], list[dict[str: float]]]:
    epoch_fractions = []
    metrics_by_fraction = []
    epochs = get_saved_epochs(net_type=net_type, identifier=identifier)
    for e in epochs:
        batches = get_saved_batches(net_type=net_type, identifier=identifier, epoch=e)
        last_batch = batches[-1]
        for b in batches:
            metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
            if metrics is not None:
                epoch_fraction = (e - 1) + (b + 1) / (last_batch + 1)
                epoch_fractions.append(epoch_fraction)
                metrics_by_fraction.append(metrics)

    return epoch_fractions, metrics_by_fraction


def load_metrics_by_batch(net_type: str, identifier: str) -> tuple[list[int], list[dict[str: float]]]:
    cumulative_batches = []
    metrics_by_cum_batch = []
    epochs = get_saved_epochs(net_type=net_type, identifier=identifier)
    previous_epoch_last_batch = 0
    for e in epochs:
        batches = get_saved_batches(net_type=net_type, identifier=identifier, epoch=e)
        for b in batches:
            metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
            if metrics is not None:
                cum_batch = previous_epoch_last_batch + (b + 1)
                cumulative_batches.append(cum_batch)
                metrics_by_cum_batch.append(metrics)

        previous_epoch_last_batch += batches[-1]

    return cumulative_batches, metrics_by_cum_batch


def precision_by_class(tp: dict, fp: dict, print_precisions=False):
    # print precision for each class:
    class_prec = {}
    for c in tp.keys():
        if tp[c] + fp[c] != 0:
            precision = tp[c] / (tp[c] + fp[c])
            class_prec[c] = precision
            if print_precisions:
                print(f'Precision for class: {c} is {100 * precision:.2f} %')
        else:
            class_prec[c] = "nan"
            if print_precisions:
                print(f'Precision for class: {c} is nan')
    return class_prec


def recall_by_class(tp: dict, fn: dict, print_recalls=False):
    # print recall for each class:
    class_recall = {}
    for c in tp.keys():
        if (tp[c] + fn[c]) != 0:
            recall = tp[c] / (tp[c] + fn[c])
            class_recall[c] = recall
            if print_recalls:
                print(f'Recall for class: {c} is {100 * recall:.2f} %')
        else:
            class_recall[c] = "nan"
            if print_recalls:
                print(f'Recall for class: {c} is nan')
    return class_recall


def specificity_by_class(tn: dict, fp: dict, print_specificity=False):
    # print specificity for each class:
    class_specificity = {}
    for c in tn.keys():
        if (tn[c] + fp[c]) != 0:
            spec = tn[c] / (tn[c] + fp[c])
            class_specificity[c] = spec
            if print_specificity:
                print(f'Specificity for class: {c} is {100 * spec:.2f} %')
        else:
            class_specificity[c] = "nan"
            if print_specificity:
                print(f'Specificity for class: {c} is nan')

    return class_specificity


def accuracy_by_class(tp: dict, tn: dict, fp: dict, fn: dict, print_accuracies=False):
    # print accuracy for each class:
    class_acc = {}
    for c in tp.keys():
        accuracy = (tp[c] + tn[c]) / (tp[c] + tn[c] + fp[c] + fn[c])
        class_acc[c] = accuracy
        if print_accuracies:
            print(f'Accuracy for class: {c} is {100 * accuracy:.2f} %')
    return class_acc


def f1_by_class(tp: dict, fp: dict, fn: dict, print_f1s=False):
    # print F1 for each class:
    class_f1 = {}
    precision = precision_by_class(tp, fp, print_precisions=False)
    recall = recall_by_class(tp, fn, print_recalls=False)
    for c in tp.keys():
        if precision[c] == "nan" or recall[c] == "nan":
            class_f1[c] = "nan"
            if print_f1s:
                print(f'F1 for class: {c} is nan')
        elif (precision[c] + recall[c]) == 0:
            class_f1[c] = 0
            if print_f1s:
                print(f'F1 for class: {c} is 0')
        else:
            f1 = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
            class_f1[c] = f1
            if print_f1s:
                print(f'F1 for class: {c} is {100 * f1:.2f} %')
    return class_f1


def micro_average_precision(tp: dict, fp: dict, print_precision=False) -> float:
    num = 0
    den = 0
    for c in tp.keys():
        num += tp[c]
        den += tp[c] + fp[c]

    precision = num / den
    if print_precision:
        print(f'Micro Average Precision: {100 * precision:.2f} %')
    return precision


def micro_average_recall(tp: dict, fn: dict, print_recall=False) -> float:
    # print precision for each class:
    num = 0
    den = 0
    for c in tp.keys():
        num += tp[c]
        den += tp[c] + fn[c]

    recall = num / den
    if print_recall:
        print(f'Micro Average Recall: {100 * recall:.2f} %')
    return recall


def micro_average_accuracy(tp: dict, tn: dict, fp: dict, fn: dict, print_accuracy=False) -> float:
    num = 0
    den = 0
    for c in tp.keys():
        num += (tp[c] + tn[c])
        den += (tp[c] + tn[c] + fp[c] + fn[c])

    micro_acc = num / den
    if print_accuracy:
        print(f'Micro Average accuracy: {100 * micro_acc:.2f} %')
    return micro_acc


def micro_average_specificity(tn: dict, fp: dict, print_specificity=False) -> float:
    # print precision for each class:
    num = 0
    den = 0
    for c in tn.keys():
        num += tn[c]
        den += tn[c] + fp[c]

    micro_spec = num / den
    if print_specificity:
        print(f'Micro Average Specificity: {100 * micro_spec:.2f} %')
    return micro_spec


def micro_average_f1(tp: dict, fp: dict, fn: dict, print_f1=False) -> float:
    micro_prec = micro_average_precision(tp, fp, print_precision=False)
    micro_rec = micro_average_recall(tp, fn, print_recall=False)
    if micro_rec + micro_rec == 0:
        micro_f1 = 0
        if print_f1:
            print(f'Micro Average F1: 0')
    else:
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if print_f1:
            print(f'Micro Average F1: {100 * micro_f1:.2f} %')
    return micro_f1


def macro_average_precision(tp: dict, fp: dict, print_precision=False) -> float:
    precisions = precision_by_class(tp, fp, print_precisions=False)

    vals = [val for val in precisions.values() if val != "nan"]
    macro_prec = float(np.mean(vals))
    if print_precision:
        print(f'Macro Average Precision: {100 * macro_prec:.2f} %')
    return macro_prec


def macro_average_recall(tp: dict, fn: dict, print_recall=False) -> float:
    recalls = recall_by_class(tp, fn, print_recalls=False)
    vals = [val for val in recalls.values() if val != "nan"]
    macro_rec = float(np.mean(vals))
    if print_recall:
        print(f'Macro Average Recall: {100 * macro_rec:.2f} %')
    return macro_rec


def macro_average_specificity(tn: dict, fp: dict, print_specificity=False) -> float:
    specs = specificity_by_class(tn, fp, print_specificity=False)
    vals = [val for val in specs.values() if val != "nan"]
    macro_spec = float(np.mean(vals))
    if print_specificity:
        print(f'Macro Average Specificity: {100 * macro_spec:.2f} %')
    return macro_spec


def macro_average_accuracy(tp: dict, tn: dict, fp: dict, fn: dict, print_accuracy=False) -> float:
    class_accs = accuracy_by_class(tp, tn, fp, fn, print_accuracies=False)

    vals = [val for val in class_accs.values() if val != "nan"]
    macro_acc = float(np.mean(vals))
    if print_accuracy:
        print(f'Macro Average accuracy: {100 * macro_acc:.2f} %')
    return macro_acc


def macro_average_f1(tp: dict, fp: dict, fn: dict, print_f1=False) -> float:
    macro_prec = macro_average_precision(tp, fp, print_precision=False)
    macro_rec = macro_average_recall(tp, fn, print_recall=False)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)
    if print_f1:
        print(f'Macro Average F1: {100 * macro_f1:.2f} %')
    return macro_f1


def get_window_stats(window_labels: torch.tensor, window_predictions: torch.tensor,
                     class_names=("normal", "central_apnea", "obstructive_apnea", "hypopnea", "spO2_desat")) \
        -> tuple[int, int, dict[str:int], dict[str:int], dict[str:int], dict[str:int]]:
    total_pred_win = 0
    correct_pred_win = 0
    tp_win = {c: 0 for c in class_names}
    tn_win = {c: 0 for c in class_names}
    fp_win = {c: 0 for c in class_names}
    fn_win = {c: 0 for c in class_names}

    tp_win2 = {c: 0 for c in class_names}
    tn_win2 = {c: 0 for c in class_names}
    fp_win2 = {c: 0 for c in class_names}
    fn_win2 = {c: 0 for c in class_names}

    total_pred_win2 = len(window_predictions)

    correct_mask = window_labels == window_predictions
    wrong_mask = ~ correct_mask
    correct_pred_win2 = sum(correct_mask)
    for c, name in enumerate(class_names):
        # True prediction:
        tp_win2[name] += sum(window_labels[correct_mask] == c)

        # False prediction:
        fp_win2[name] += sum(window_predictions[wrong_mask] == c)
        fn_win2[name] += sum(window_labels[wrong_mask] == c)

    for c in class_names:
        tn_win2[c] = total_pred_win2 - tp_win2[c] - fp_win2[c] - fn_win2[c]
    #
    # # Original slower method:
    # for wi in range(len(window_labels)):
    #     label = window_labels[wi]
    #     prediction = window_predictions[wi]
    #     true_class_name = class_names[label]
    #     pred_class_name = class_names[prediction]
    #     total_pred_win += 1
    #
    #     if label == prediction:
    #         # True prediction:
    #         correct_pred_win += 1
    #         tp_win[true_class_name] += 1
    #         for c in class_names:
    #             if c != true_class_name:
    #                 tn_win[c] += 1
    #     else:
    #         # False prediction:
    #         for c in class_names:
    #             if c == pred_class_name:
    #                 fp_win[c] += 1
    #             elif c == true_class_name:
    #                 fn_win[c] += 1
    #             else:
    #                 tn_win[c] += 1
    #
    # assert tp_win == tp_win2
    # assert tn_win == tn_win2
    # assert fp_win == fp_win2
    # assert fn_win == fn_win2
    # assert correct_pred_win == correct_pred_win
    # assert total_pred_win == total_pred_win2

    return total_pred_win, correct_pred_win, tp_win, tn_win, fp_win, fn_win


def get_window_label(window_labels: torch.tensor):
    if torch.sum(window_labels) == 0:
        return torch.tensor(0, dtype=torch.int64), 1.0
    else:
        # 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation
        unique_events, event_counts = window_labels.unique(
            return_counts=True)  # Tensor containing counts of unique values.
        prominent_event_index = torch.argmax(event_counts)
        prominent_event = unique_events[prominent_event_index]
        confidence = event_counts[prominent_event] / torch.numel(window_labels)

        # print(event_counts)
        return prominent_event.type(torch.int64), float(confidence)


def get_window_stats_new(window_labels: torch.tensor, window_predictions: torch.tensor):
    classes = ("normal", "central_apnea", "obstructive_apnea", "hypopnea", "spO2_desat")
    cm_win = confusion_matrix(target=window_labels, preds=window_predictions, num_classes=5, task="multiclass")

    total_pred_win = torch.sum(cm_win)
    tps = torch.diag(cm_win)
    fps = torch.sum(cm_win, dim=0, keepdim=False) - tps
    fns = torch.sum(cm_win, dim=1, keepdim=False) - tps
    tns = total_pred_win - (fps + fns + tps)
    correct_pred_win = torch.sum(tps)

    tp_win = {}
    tn_win = {}
    fp_win = {}
    fn_win = {}
    for c, name in enumerate(classes):
        tp_win[name] = tps[c].item()
        tn_win[name] = tns[c].item()
        fp_win[name] = fps[c].item()
        fn_win[name] = fns[c].item()

    return cm_win, total_pred_win.item(), correct_pred_win.item(), tp_win, tn_win, fp_win, fn_win


def test_loop(model: nn.Module, test_dataloader: DataLoader, device="cpu", max_batches=None,
              progress_bar=True, verbose=False, first_batch=0) \
        -> tuple[dict[str: float], list[list[float]], dict[str: dict[str: list | float]]]:
    if verbose:
        print(datetime.datetime.now())
    loader = test_dataloader
    batches = len(loader)

    if max_batches is not None and max_batches < batches:
        batches = max_batches

    # print(f"Batches in epoch: {batches}")

    unix_time_start = time.time()

    # prepare to count predictions for each class
    classes = ("normal", "central_apnea", "obstructive_apnea", "hypopnea", "spO2_desat")
    thresholds = torch.tensor(np.linspace(start=0, stop=1, num=500), device=device)
    tprs_by_class = {c: [] for c in classes}
    fprs_by_class = {c: [] for c in classes}
    aucs_by_class = {c: [] for c in classes}

    cm = torch.zeros((5, 5), device=device)
    correct_pred = 0
    total_pred = 0
    tp = {c: 0 for c in classes}
    tn = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    # Switch to eval mode:
    model.eval()
    model = model.to(device)

    if progress_bar:
        pbar_test_loop = tqdm(initial=first_batch, total=batches, leave=False, desc="Test loop")
    else:
        pbar_test_loop = None

    with torch.no_grad():
        for (batch_i, data) in enumerate(loader, start=first_batch):
            # get the inputs; data is a list of [inputs, labels]
            batch_inputs, batch_labels = data

            # Convert to accepted dtypes: float32, float64, int64 and maybe more but not sure
            batch_labels = batch_labels.type(torch.int64)

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # Predictions:
            batch_outputs = model(batch_inputs)
            batch_output_probs = F.softmax(batch_outputs, dim=1)
            _, batch_predictions = torch.max(batch_outputs, dim=1, keepdim=False)

            # Adjust labels in case of by window classification
            if batch_outputs.numel() != batch_inputs.numel() * 5 and batch_labels.shape[0] != batch_labels.numel():
                # Per window classification nut labels per sample:
                labels_by_window = torch.zeros((batch_labels.shape[0]), device=device, dtype=torch.int64)
                for batch_j in range(batch_labels.shape[0]):
                    labels_by_window[batch_j] = get_window_label(batch_labels[batch_j, :])[0].to(device)
                batch_labels = labels_by_window

            # Compute RoC:
            roc = ROC(task="multiclass", thresholds=thresholds, num_classes=5).to(device)
            auroc = AUROC(task="multiclass", thresholds=thresholds, num_classes=5, average="none").to(device)
            fprs, tprs, _ = roc(batch_output_probs, batch_labels)
            aucs = auroc(batch_output_probs, batch_labels)
            for c in range(len(classes)):
                class_name = classes[c]
                fprs_by_class[class_name].append(fprs[c, :])
                tprs_by_class[class_name].append(tprs[c, :])
                aucs_by_class[class_name].append(aucs[c])

            # collect the correct predictions for each class
            if device == "cpu":
                with Pool(NUM_PROCESSES_FOR_METRICS) as pool:
                    batch_labels_preds = zip(batch_labels, batch_predictions)
                    for result in pool.starmap(get_window_stats_new, batch_labels_preds):
                        # total_pred_win, correct_pred_win, tp_win, tn_win, fp_win, fn_win = result
                        cm_win, total_pred_win, correct_pred_win, tp_win, tn_win, fp_win, fn_win = result
                        cm += cm_win
                        total_pred += total_pred_win
                        correct_pred += correct_pred_win
                        for c in classes:
                            tp[c] += tp_win[c]
                            tn[c] += tn_win[c]
                            fp[c] += fp_win[c]
                            fn[c] += fn_win[c]
            else:
                batch_labels = torch.ravel(batch_labels)
                batch_predictions = torch.ravel(batch_predictions)

                result = get_window_stats_new(batch_labels, batch_predictions)
                cm_win, total_pred_win, correct_pred_win, tp_win, tn_win, fp_win, fn_win = result

                cm += cm_win
                total_pred += total_pred_win
                correct_pred += correct_pred_win
                for c in classes:
                    tp[c] += tp_win[c]
                    tn[c] += tn_win[c]
                    fp[c] += fp_win[c]
                    fn[c] += fn_win[c]

            # print statistics
            if verbose and batch_i % 100 == 99:  # print every 10000 mini-batches
                time_elapsed = time.time() - unix_time_start

                print(f'[Batch: {batch_i + 1:7d}/{batches:7d}]'
                      f' Secs/Batch: {time_elapsed / (batch_i + 1)}:.2f')

            if progress_bar:
                pbar_test_loop.update(1)

            if (batch_i + 1) >= batches:
                break

    if progress_bar:
        pbar_test_loop.close()

    # Compute threshold average ROC for each class:
    roc_info_by_class = {}
    average_auc_by_class = {}
    for class_name in classes:
        fprs = torch.stack(fprs_by_class[class_name], dim=0)
        average_fpr = torch.mean(fprs, dim=0, keepdim=False)
        tprs = torch.stack(tprs_by_class[class_name], dim=0)
        average_tpr = torch.mean(tprs, dim=0, keepdim=False)
        aucs = torch.tensor(aucs_by_class[class_name])
        average_auc = torch.mean(aucs)

        average_auc_by_class[class_name] = average_auc.item()

        roc_info_by_class[class_name] = {
            "thresholds": thresholds.tolist(),
            "average_fpr": average_fpr.tolist(),
            "average_tpr": average_tpr.tolist(),
            "average_auc": average_auc.item()
        }

    aggregate_acc = correct_pred / total_pred
    if verbose:
        print(f"Intuitive aggregate accuracy: {100 * aggregate_acc:.2f}%")

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

    # Note: In multiclass classification with symmetric costs, the micro average precision, recall,
    # and aggregate accuracy scores are mathematically equivalent because the sum of fp and sum of fn are equal.
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
               "specificity_by_class": spec_by_class,
               "average_auc_by_class": average_auc_by_class}

    if verbose:
        print('Finished Testing')
        print(datetime.datetime.now())

    return metrics, cm.tolist(), roc_info_by_class



def test_all_checkpoints(net_type: str, identifier: str, test_dataloader: DataLoader, device=COMPUTE_PLATFORM,
                         max_batches=MAX_BATCHES, progress_bar=True):
    if LOAD_FROM_BATCH > 0:
        print("WARNING: Loading from test batch other than the first (0) is not supported by test_all_checkpoints! "
              "Setting it to 0")
        test_dataloader.sampler.first_batch_index = 0

    pbar1 = None
    pbar2 = None
    epochs = sorted(get_saved_epochs(net_type=net_type, identifier=identifier), reverse=True)
    if progress_bar:
        pbar1 = tqdm(total=len(epochs), desc="Epoch checkpoint", leave=True)
    for e in epochs:
        batches = sorted(get_saved_batches(net_type=net_type, identifier=identifier, epoch=e), reverse=True)
        if progress_bar:
            pbar2 = tqdm(total=len(batches), desc="Batch checkpoint", leave=False)
        for b in batches:
            metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
            if metrics is None:
                net, _, _, _, _, _, _, _, _, _ = load_checkpoint(net_type=net_type, identifier=identifier, epoch=e,
                                                                 batch=b,
                                                                 device=device)
                metrics, cm, roc_info = test_loop(model=net, test_dataloader=test_dataloader, device=device,
                                                  max_batches=max_batches,
                                                  progress_bar=progress_bar, verbose=False)
                save_metrics(metrics=metrics, net_type=net_type, identifier=identifier, epoch=e, batch=b)
                save_confusion_matrix(confusion_matrix=cm, net_type=net_type, identifier=identifier, epoch=e,
                                      batch=b)
                save_rocs(roc_info, net_type=net_type, identifier=identifier, epoch=e, batch=b, save_plot=True)

            if pbar2 is not None:
                pbar2.update(1)
        if progress_bar:
            pbar2.close()
            pbar1.update(1)
    if progress_bar:
        pbar1.close()


def test_all_epochs(net_type: str, identifier: str, test_dataloader: DataLoader, device=COMPUTE_PLATFORM,
                    max_batches=MAX_BATCHES, progress_bar=True):
    if LOAD_FROM_BATCH > 0:
        print("WARNING: Loading from test batch other than the first (0) is not supported by test_all_epochs! "
              "Setting it to 0")
        test_dataloader.sampler.first_batch_index = 0

    pbar1 = None
    epochs = sorted(get_saved_epochs(net_type=net_type, identifier=identifier), reverse=True)
    if progress_bar:
        pbar1 = tqdm(total=len(epochs), desc="Epoch checkpoint", leave=True)

    for e in epochs:
        b = get_last_batch(net_type=net_type, identifier=identifier, epoch=e)
        metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
        if metrics is None or OVERWRITE_METRICS:

            net, _, _, _, _, _, _, _, _, _ = load_checkpoint(net_type=net_type, identifier=identifier, epoch=e, batch=b,
                                                             device=device)

            metrics, cm, roc_info = test_loop(model=net, test_dataloader=test_dataloader, device=device,
                                              max_batches=max_batches,
                                              progress_bar=progress_bar, verbose=False)

            save_metrics(metrics=metrics, net_type=net_type, identifier=identifier, epoch=e,
                         batch=b)
            save_confusion_matrix(confusion_matrix=cm, net_type=net_type, identifier=identifier, epoch=e,
                                  batch=b)
            save_rocs(roc_info, net_type=net_type, identifier=identifier, epoch=e, batch=b, save_plot=True)
        if progress_bar:
            pbar1.update(1)
    if progress_bar:
        pbar1.close()


def test_last_checkpoint(net_type: str, identifier: str, test_dataloader: DataLoader, device=COMPUTE_PLATFORM,
                         max_batches=MAX_BATCHES, first_batch=0, progress_bar=True):
    test_dataloader.sampler.first_batch_index = first_batch
    e = get_last_epoch(net_type=net_type, identifier=identifier)
    b = get_last_batch(net_type=net_type, identifier=identifier, epoch=e)
    metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
    if metrics is None:
        net, _, _, _, _, _, _, _, _, _ = load_checkpoint(net_type=net_type, identifier=identifier, epoch=e, batch=b,
                                                         device=device)
        metrics, cm, roc_info = test_loop(model=net, test_dataloader=test_dataloader, device=device,
                                          first_batch=first_batch, max_batches=max_batches,
                                          progress_bar=progress_bar, verbose=False)
        save_metrics(metrics=metrics, net_type=net_type, identifier=identifier, epoch=e, batch=b)
        save_confusion_matrix(confusion_matrix=cm, net_type=net_type, identifier=identifier, epoch=e,
                              batch=b)
        save_rocs(roc_info, net_type=net_type, identifier=identifier, epoch=e, batch=b, save_plot=True)


if __name__ == "__main__":
    if COMPUTE_PLATFORM == "opencl":
        PATH_TO_PT_OCL_DLL = Path(config["paths"]["local"]["pt_ocl_dll"])
        PATH_TO_DEPENDENCY_DLLS = Path(config["paths"]["local"]["dependency_dlls"])
        os.add_dll_directory(str(PATH_TO_DEPENDENCY_DLLS))
        torch.ops.load_library(str(PATH_TO_PT_OCL_DLL))
        test_device = "privateuseone:0"
    elif torch.cuda.is_available():
        test_device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        test_device = torch.device("mps")
    else:
        test_device = torch.device("cpu")

    test_loader = get_pre_batched_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS,
                                              pre_fetch=PRE_FETCH,
                                              shuffle=False)
    if TEST_MODEL:
        print(f"Device: {test_device}")
        print(NET_TYPE)
        print(IDENTIFIER)
        # test_loader = data_loaders_mapped.get_saved_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=2,
        # pre_fetch=1, shuffle=False, use_existing_batch_indices=True)
        test_loader.sampler.first_batch_index = LOAD_FROM_BATCH
        # test_all_checkpoints(net_type=NET_TYPE, identifier=IDENTIFIER, test_dataloader=test_loader,
        # device=test_device, max_batches=MAX_BATCHES, progress_bar=True)
        test_all_epochs(net_type=NET_TYPE, identifier=IDENTIFIER, test_dataloader=test_loader, device=test_device,
                        max_batches=MAX_BATCHES, progress_bar=True)

    # e = get_last_epoch(net_type=NET_TYPE, identifier=IDENTIFIER)
    # b = get_last_batch(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=e)
    # net, _, _, _, _, _, _, _, _, _ = load_checkpoint(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=e, batch=b,
    #                                                  device=test_device)
    # plot_sample_prediction_sequence(model=net, test_dataloader=test_loader, device=test_device, n_batches=1)

    epoch_frac, metrics = load_metrics_by_epoch(net_type=NET_TYPE, identifier=IDENTIFIER)
    accuracies = [m["aggregate_accuracy"] for m in metrics]
    micro_accuracies = [m["micro_accuracy"] for m in metrics if "micro_accuracy" in m.keys()]
    macro_accuracies = [m["macro_accuracy"] for m in metrics if "macro_accuracy" in m.keys()]
    macro_precisions = [m["macro_precision"] for m in metrics]
    macro_recalls = [m["macro_recall"] for m in metrics]
    macro_f1s = [m["macro_f1"] for m in metrics]

    # fig, axis = plt.subplots(4, 2)
    plt.figure()
    plt.plot(epoch_frac, accuracies, label="accuracy")
    if len(micro_accuracies) == len(macro_accuracies) == len(epoch_frac):
        plt.plot(epoch_frac, micro_accuracies, label="micro_accuracies")
        plt.plot(epoch_frac, macro_accuracies, label="macro_accuracies")
    plt.plot(epoch_frac, macro_precisions, label="macro_precision")
    plt.plot(epoch_frac, macro_recalls, label="macro_recall")
    plt.plot(epoch_frac, macro_f1s, label="macro_f1")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Model ({NET_TYPE}) test performance over epochs")
    plt.show()
