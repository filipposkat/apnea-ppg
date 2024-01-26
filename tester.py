import yaml
from pathlib import Path
import datetime, time
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import data_loaders_mapped

# Local imports:
from data_loaders_mapped import MappedDataset, BatchSampler
from pre_batched_dataloader import get_pre_batched_test_loader, get_pre_batched_test_cross_sub_loader
from UNet import UNet

if __name__ == "__main__":
    from trainer import get_saved_epochs, get_saved_batches, get_last_batch, get_last_epoch, load_checkpoint

# --- START OF CONSTANTS --- #
NET_TYPE: str = "UResIncNet"  # UNET or UResIncNet
IDENTIFIER: str = "ks3-depth8-strided-0"
EPOCHS = 100
BATCH_SIZE_TEST = 8192
MAX_BATCHES = None  # Maximum number of test batches to use or None to use all of them
LOAD_FROM_BATCH = 0
NUM_WORKERS = 2
PRE_FETCH = 2
DEVICE = "cpu"

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
    if "subset_1_saved_models_directory" in config:
        MODELS_PATH = Path(config["paths"]["local"]["subset_1_saved_models_directory"])
    else:
        MODELS_PATH = PATH_TO_SUBSET1_TRAINING.joinpath("saved-models")
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1
    MODELS_PATH = PATH_TO_SUBSET1_TRAINING.joinpath("saved-models")

MODELS_PATH.mkdir(parents=True, exist_ok=True)


# --- END OF CONSTANTS --- #

def save_metrics(metrics: dict[str: float], net_type: str, identifier: str, epoch: int, batch: int):
    metrics_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_metrics.json")

    with open(metrics_path, 'w') as file:
        json.dump(metrics, file)


def load_metrics(net_type: str, identifier: str, epoch: int, batch: int) -> dict[str: float]:
    metrics_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}-test_metrics.json")
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
        if precision[c] != "nan" and recall[c] != "nan":
            f1 = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
            class_f1[c] = f1
            if print_f1s:
                print(f'F1 for class: {c} is {100 * f1:.2f} %')
        else:
            class_f1[c] = "nan"
            if print_f1s:
                print(f'F1 for class: {c} is nan')
    return class_f1


def micro_average_precision(tp: dict, fp: dict, print_precision=False):
    # print precision for each class:
    num = 0
    den = 0
    for c in tp.keys():
        num += tp[c]
        den += tp[c] + fp[c]

    precision = num / den
    if print_precision:
        print(f'Micro Average Precision: {100 * precision:.2f} %')
    return precision


def micro_average_recall(tp: dict, fn: dict, print_recall=False):
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


def micro_average_f1(tp: dict, fp: dict, fn: dict, print_f1=False):
    micro_prec = micro_average_precision(tp, fp, print_precision=False)
    micro_rec = micro_average_recall(tp, fn, print_recall=False)
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    if print_f1:
        print(f'Micro Average F1: {100 * micro_f1:.2f} %')
    return micro_f1


def macro_average_precision(tp: dict, fp: dict, print_precision=False):
    precisions = precision_by_class(tp, fp, print_precisions=False)

    vals = [val for val in precisions.values() if val != "nan"]
    macro_prec = np.mean(vals)
    if print_precision:
        print(f'Macro Average Precision: {100 * macro_prec:.2f} %')
    return macro_prec


def macro_average_recall(tp: dict, fn: dict, print_recall=False):
    recalls = recall_by_class(tp, fn, print_recalls=False)
    vals = [val for val in recalls.values() if val != "nan"]
    macro_rec = np.mean(vals)
    if print_recall:
        print(f'Macro Average Recall: {100 * macro_rec:.2f} %')
    return macro_rec


def macro_average_f1(tp: dict, fp: dict, fn: dict, print_f1=False):
    macro_prec = macro_average_precision(tp, fp, print_precision=False)
    macro_rec = macro_average_recall(tp, fn, print_recall=False)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)
    if print_f1:
        print(f'Macro Average F1: {100 * macro_f1:.2f} %')
    return macro_f1


def test_loop(net: nn.Module, test_loader: DataLoader, device="cpu", max_batches=None,
              progress_bar=True, verbose=False, first_batch=0) -> dict[str: float]:
    if verbose:
        print(datetime.datetime.now())
    loader = test_loader
    batches = len(loader)

    if max_batches is not None and max_batches < batches:
        batches = max_batches

    # print(f"Batches in epoch: {batches}")

    unix_time_start = time.time()

    # prepare to count predictions for each class
    classes = ("normal", "central_apnea", "obstructive_apnea", "hypopnea", "spO2_desat")
    correct_pred = 0
    total_pred = 0
    tp = {c: 0 for c in classes}
    tn = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    # Switch to eval mode:
    net.eval()
    net.to(device)

    if progress_bar:
        pbar_test_loop = tqdm(initial=first_batch, total=batches, leave=False, desc="Test loop")
    else:
        pbar_test_loop = None

    with torch.no_grad():
        for (batch_i, data) in enumerate(loader, start=first_batch):
            # get the inputs; data is a list of [inputs, labels]
            batch_inputs, batch_labels = data
            batch_inputs.to(device)

            # Predictions:
            batch_outputs = net(batch_inputs)
            _, batch_predictions = torch.max(batch_outputs, dim=1)

            batch_predictions.to("cpu")

            # collect the correct predictions for each class
            for window_labels, window_predictions in zip(batch_labels, batch_predictions):
                for wi in range(len(window_labels)):
                    label = window_labels[wi]
                    prediction = window_predictions[wi]
                    true_class_name = classes[label]
                    pred_class_name = classes[prediction]
                    total_pred += 1

                    if label == prediction:
                        # True prediction:
                        correct_pred += 1
                        tp[true_class_name] += 1
                        for c in classes:
                            if c != true_class_name:
                                tn[c] += 1
                    else:
                        # False prediction:
                        for c in classes:
                            if c == pred_class_name:
                                fp[c] += 1
                            elif c == true_class_name:
                                fn[c] += 1
                            else:
                                tn[c] += 1

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

    aggregate_acc = correct_pred / total_pred
    if verbose:
        print(f"Intuitive aggregate accuracy: {100 * aggregate_acc:.2f}%")

    acc_by_class = accuracy_by_class(tp, tn, fp, fn, print_accuracies=verbose)
    prec_by_class = precision_by_class(tp, fp, print_precisions=verbose)
    rec_by_class = recall_by_class(tp, fn, print_recalls=verbose)
    sepc_by_class = specificity_by_class(tn, fp, print_specificity=verbose)
    f1_per_class = f1_by_class(tp, fp, fn, print_f1s=verbose)
    micro_prec = micro_average_precision(tp, fp, print_precision=verbose)
    micro_rec = micro_average_recall(tp, fn, print_recall=verbose)
    micro_f1 = micro_average_f1(tp, fp, fn, print_f1=verbose)

    macro_prec = macro_average_precision(tp, fp, print_precision=verbose)
    macro_rec = macro_average_recall(tp, fn, print_recall=verbose)
    macro_f1 = macro_average_f1(tp, fp, fn, print_f1=verbose)

    metrics = {"aggregate_accuracy": aggregate_acc,
               "macro_precision": macro_prec,
               "macro_recall": macro_rec,
               "macro_f1": macro_f1,
               "micro_precision": micro_prec,
               "micro_recall": micro_rec,
               "micro_f1": micro_f1,
               "accuracy_by_class": acc_by_class,
               "precision_by_class": prec_by_class,
               "recall_by_class": rec_by_class,
               "f1_by_class": f1_per_class,
               "specificity_by_class": sepc_by_class}

    if verbose:
        print('Finished Testing')
        print(datetime.datetime.now())

    return metrics


def test_all_checkpoints(net_type: str, identifier: str, test_loader: DataLoader, device=DEVICE,
                         max_batches=MAX_BATCHES, progress_bar=True):
    if LOAD_FROM_BATCH > 0:
        print("WARNING: Loading from test batch other than the first (0) is not supported by test_all_checkpoints! "
              "Setting it to 0")
        test_loader.sampler.first_batch_index = 0

    pbar1 = None
    pbar2 = None
    epochs = sorted(get_saved_epochs(net_type=net_type, identifier=identifier), reverse=True)
    if progress_bar:
        pbar1 = tqdm(total=len(epochs), desc="Epoch checkpoint", leave=True)
    for e in epochs:
        batches = sorted(get_saved_batches(net_type=net_type, identifier=identifier, epoch=e), reverse=True)
        if progress_bar:
            pbar2 = tqdm(total=len(batches), desc="Train checkpoint", leave=True)
        for b in batches:
            metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
            if metrics is None:
                net, _, _, _, _, _ = load_checkpoint(net_type=net_type, identifier=identifier, epoch=e, batch=b)
                metrics = test_loop(net=net, test_loader=test_loader, device=device, max_batches=max_batches,
                                    progress_bar=progress_bar, verbose=False)
                save_metrics(metrics=metrics, net_type=net_type, identifier=identifier, epoch=e, batch=b)
            if pbar2 is not None:
                pbar2.update(1)
        if progress_bar:
            pbar2.close()
            pbar1.update(1)
    if progress_bar:
        pbar1.close()


def test_all_epochs(net_type: str, identifier: str, test_loader: DataLoader, device=DEVICE,
                    max_batches=MAX_BATCHES, progress_bar=True):
    if LOAD_FROM_BATCH > 0:
        print("WARNING: Loading from test batch other than the first (0) is not supported by test_all_epochs! "
              "Setting it to 0")
        test_loader.sampler.first_batch_index = 0

    pbar1 = None
    epochs = sorted(get_saved_epochs(net_type=net_type, identifier=identifier), reverse=True)
    if progress_bar:
        pbar1 = tqdm(total=len(epochs), desc="Epoch checkpoint", leave=True)
    for e in epochs:
        b = get_last_batch(net_type=net_type, identifier=identifier, epoch=e)
        metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
        if metrics is None:
            net, _, _, _, _, _ = load_checkpoint(net_type=net_type, identifier=identifier, epoch=e, batch=b)
            metrics = test_loop(net=net, test_loader=test_loader, device=device, max_batches=max_batches,
                                progress_bar=progress_bar, verbose=False)
            save_metrics(metrics=metrics, net_type=net_type, identifier=identifier, epoch=e, batch=b)
        if progress_bar:
            pbar1.update(1)
    if progress_bar:
        pbar1.close()


def test_last_checkpoint(net_type: str, identifier: str, test_loader: DataLoader, device=DEVICE,
                         max_batches=MAX_BATCHES, first_batch=0, progress_bar=True):
    test_loader.sampler.first_batch_index = first_batch
    e = get_last_epoch(net_type=net_type, identifier=identifier)
    b = get_last_batch(net_type=net_type, identifier=identifier, epoch=e)
    metrics = load_metrics(net_type=net_type, identifier=identifier, epoch=e, batch=b)
    if metrics is None:
        net, _, _, _, _, _ = load_checkpoint(net_type=net_type, identifier=identifier, epoch=e, batch=b)
        metrics = test_loop(net=net, test_loader=test_loader, device=device,
                            first_batch=first_batch, max_batches=max_batches,
                            progress_bar=progress_bar, verbose=False)
        save_metrics(metrics=metrics, net_type=net_type, identifier=identifier, epoch=e, batch=b)


if __name__ == "__main__":
    test_loader = get_pre_batched_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=2, pre_fetch=1, shuffle=False)
    # test_loader = data_loaders_mapped.get_saved_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=2, pre_fetch=1,
    #                                                         shuffle=False, use_existing_batch_indices=True)
    test_loader.sampler.first_batch_index = LOAD_FROM_BATCH
    test_all_checkpoints(net_type=NET_TYPE, identifier=IDENTIFIER, test_loader=test_loader, device=DEVICE,
                         max_batches=MAX_BATCHES, progress_bar=True)
