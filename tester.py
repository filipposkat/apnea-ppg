import yaml
from pathlib import Path
import datetime, time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Local imports:
from data_loaders_iterable import IterDataset, get_saved_test_loader
from UNet import UNet

# --- START OF CONSTANTS --- #
EPOCHS = 100
BATCH_SIZE_TEST = 32768
LR_TO_BATCH_RATIO = 1 / 25600
LR_WARMUP = True

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1


# --- END OF CONSTANTS --- #

def precision_by_class(tp: dict, fp: dict, print_precisions=False):
    # print precision for each class:
    class_prec = {}
    for c in tp.keys():
        precision = tp[c] / (tp[c] + fp[c])
        class_prec[c] = precision
        if print_precisions:
            print(f'Precision for class: {c} is {100 * precision:.2f} %')
    return class_prec


def recall_by_class(tp: dict, fn: dict, print_recalls=False):
    # print recall for each class:
    class_recall = {}
    for c in tp.keys():
        recall = tp[c] / (tp[c] + fn[c])
        class_recall[c] = recall
        if print_recalls:
            print(f'Recall for class: {c} is {100 * recall:.2f} %')
    return class_recall


def specificity_by_class(tn: dict, fp: dict, print_specificity=False):
    # print specificity for each class:
    class_specificity = {}
    for c in tn.keys():
        spec = tn[c] / (tn[c] + fp[c])
        class_specificity[c] = spec
        if print_specificity:
            print(f'Specificity for class: {c} is {100 * spec:.2f} %')
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
        f1 = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        class_f1[c] = f1
        if print_f1s:
            print(f'F1 for class: {c} is {100 * f1:.2f} %')
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
    macro_prec = np.mean(precisions.values())
    if print_precision:
        print(f'Macro Average Precision: {100 * macro_prec:.2f} %')
    return macro_prec


def macro_average_recall(tp: dict, fn: dict, print_recall=False):
    recalls = recall_by_class(tp, fn, print_recalls=False)
    macro_rec = np.mean(recalls.values())
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


def test_loop(net: nn.Module, test_loader: DataLoader, verbose=False):
    if verbose:
        print(datetime.datetime.now())
    loader = test_loader
    batches = len(loader)
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

    with torch.no_grad():
        for (i, data) in tqdm(enumerate(loader), total=batches):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Predictions:
            outputs = net(inputs)
            _, predictions = torch.max(outputs, dim=1)

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
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
            if verbose and i % 10000 == 99999:  # print every 10000 mini-batches
                time_elapsed = time.time() - unix_time_start

                print(f'[Batch{i + 1:7d}/{batches:7d}]'
                      f' Minutes/Batch: {time_elapsed / (i + 1) / 60}')

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

# if __name__ == "__main__":
# # Prepare train dataloader:
# train_loader = get_saved_train_loader(batch_size=BATCH_SIZE_TEST)
#
# # Create Network:
# unet = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=3, sampling_method="pooling")
# unet = unet.to(device)
#
# # Define loss and optimizer:
# ce_loss = nn.CrossEntropyLoss()
# lr = LR_TO_BATCH_RATIO * BATCH_SIZE
# optim_kwargs = {"lr": 0.01, "momentum": 0.7}
# sgd = optim.SGD(unet.parameters(), **optim_kwargs)
# lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=sgd, start_factor=0.3, end_factor=1, total_iters=3)
#
# # Train:
# train_loop(train_loader=train_loader, net=unet, optimizer=sgd, optim_kwargs=optim_kwargs, criterion=ce_loss,
#            lr_scheduler=lr_scheduler, device=device, epochs=EPOCHS, save_model_every_epoch=True, identifier=1)
#
# # Save model:
# save_model_state(unet, optimizer=sgd, optimizer_kwargs=optim_kwargs,
#                  criterion=ce_loss, net_type="UNET", identifier=1,
#                  batch_size=BATCH_SIZE, epoch=EPOCHS)
