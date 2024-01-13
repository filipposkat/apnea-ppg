import yaml
from pathlib import Path
import datetime, time

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


def accuracy_by_class(correct_pred, total_pred, print_accuracies=False):
    # print accuracy for each class:
    class_acc = {}
    for c, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[c]
        class_acc[c] = accuracy
        if print_accuracies:
            print(f'Accuracy for class: {c} is {accuracy:.1f} %')
    return class_acc


def aggregate_accuracy(correct_pred, total_pred, print_accuracy=False):
    aggregate_correct_pred = 0
    aggregate_total_pred = 0
    for c, correct_count in correct_pred.items():
        aggregate_correct_pred += correct_count
        aggregate_total_pred += total_pred[c]

    aggregate_accuracy = 100 * float(aggregate_correct_pred) / aggregate_total_pred
    if print_accuracy:
        print(f"Aggregate accuracy: {aggregate_accuracy}")
    return aggregate_accuracy


def test_loop(net: nn.Module, test_loader: DataLoader, verbose=True):
    if verbose:
        print(datetime.datetime.now())
    loader = test_loader
    batches = len(loader)
    # print(f"Batches in epoch: {batches}")

    unix_time_start = time.time()

    # prepare to count predictions for each class
    classes = ("normal", "central_apnea", "obstructive_apnea", "hypopnea", "spO2_desat")
    correct_pred = {c: 0 for c in classes}
    total_pred = {c: 0 for c in classes}
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
                total_pred[true_class_name] += 1

                if label == prediction:
                    # True prediction:
                    correct_pred[true_class_name] += 1
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

    aggregate_acc = aggregate_accuracy(correct_pred, total_pred, print_accuracy=verbose)
    acc_by_class = accuracy_by_class(correct_pred, total_pred, print_accuracies=verbose)

    if verbose:
        print('Finished Testing')
        print(datetime.datetime.now())
    return aggregate_acc, acc_by_class


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
