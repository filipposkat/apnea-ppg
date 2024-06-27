import pickle

import numpy as np

import yaml
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from tqdm import tqdm

# Local imports:
from pre_batched_dataloader import get_pre_batched_train_loader, get_pre_batched_test_loader

from UNet import UNet, ConvNet
from UResIncNet import UResIncNet, ResIncNet
from CombinedNet import CombinedNet
from trainer import train_loop
from tester import test_loop

EPOCHS = 10
BATCH_SIZE = 256
BATCH_SIZE_TEST = 1024
NUM_WORKERS = 0
NUM_WORKERS_TEST = 0
PRE_FETCH = None
PRE_FETCH_TEST = None
CLASSIFY_PER_SAMPLE = True
NET_TYPE = "UResIncNet"
OPTIMIZER = "adam"
CLASS_WEIGHTS = None
LR_WARMUP = False
LR_WARMUP_STEP_EPOCH_INTERVAL = 1
SAVE_MODEL_EVERY_EPOCH = True
TESTING_EPOCH_INTERVAL = 1

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    subset_id = int(config["variables"]["dataset"]["subset"])
    PATH_TO_SUBSET = Path(config["paths"]["local"][f"subset_{subset_id}_directory"])
    PATH_TO_SUBSET_TRAINING = Path(config["paths"]["local"][f"subset_{subset_id}_training_directory"])
    COMPUTE_PLATFORM = config["system"]["specs"]["compute_platform"]
else:
    subset_id = 1
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_TRAINING = PATH_TO_SUBSET
    COMPUTE_PLATFORM = "cpu"


def full_train_loop(train_config: dict):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Prepare train dataloader:
    train_loader = get_pre_batched_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PRE_FETCH)
    test_loader = get_pre_batched_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS_TEST,
                                              pre_fetch=PRE_FETCH_TEST, shuffle=False)
    # Find out window size:
    sample_batch_input, sample_batch_labels = next(iter(train_loader))
    window_size = sample_batch_input.shape[2]

    # Class weights:
    weights = None
    if CLASS_WEIGHTS is not None:
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    net = None
    net_kwargs = {}
    if NET_TYPE == "UNET":
        net = UNet(nclass=5, in_chans=1, max_channels=512, depth=train_config["depth"],
                   layers=train_config["layers"],
                   kernel_size=train_config["kernel_size"],
                   sampling_method=train_config["sampling_method"])
        net_kwargs = net.get_kwargs()
    elif NET_TYPE == "UResIncNet":
        net = UResIncNet(nclass=5, in_chans=1, max_channels=512, depth=train_config["depth"],
                         layers=train_config["layers"],
                         kernel_size=train_config["kernel_size"],
                         sampling_factor=2, sampling_method=train_config["sampling_method"], skip_connection=True,
                         custom_weight_init=False)
        net_kwargs = net.get_kwargs()
    elif NET_TYPE == "ConvNet":
        net = ConvNet(nclass=5, in_size=window_size, in_chans=1, max_channels=512, depth=train_config["depth"],
                      layers=train_config["layers"], kernel_size=train_config["kernel_size"],
                      sampling_method=train_config["sampling_method"])
        net_kwargs = net.get_kwargs()
    elif NET_TYPE == "ResIncNet":
        net = ResIncNet(nclass=5, in_size=window_size, in_chans=1, max_channels=512, depth=train_config["depth"],
                        layers=train_config["layers"],
                        kernel_size=train_config["kernel_size"],
                        sampling_factor=2, sampling_method=train_config["sampling_method"], skip_connection=True)
        net_kwargs = net.get_kwargs()
    elif NET_TYPE == "CombinedNet":
        net = CombinedNet(nclass=5, in_size=window_size, in_chans=1, max_channels=512, depth=train_config["depth"],
                          kernel_size=train_config["kernel_size"], layers=train_config["layers"], sampling_factor=2,
                          sampling_method=train_config["sampling_method"],
                          skip_connection=True, lstm_max_features=train_config["lstm_hidden_size"],
                          lstm_layers=train_config["lstm_layers"],
                          lstm_dropout=train_config["lstm_dropout"], lstm_bidirectional=True,
                          lstm_depth=1, custom_weight_init=False)

    # Define loss:
    if weights is not None:
        loss_kwargs = {"weight": weights}
        loss = nn.CrossEntropyLoss(**loss_kwargs)
    else:
        loss_kwargs = None
        loss = nn.CrossEntropyLoss()

    # Model should go to device first before initializing optimizer:
    net = net.to(device)

    # Define optimizer:

    if OPTIMIZER == "adam":
        optim_kwargs = {"lr": train_config["lr"], "betas": (0.9, 0.999), "eps": 1e-08}
        optimizer = optim.Adam(net.parameters(), **optim_kwargs)
    else:  # sgd
        optim_kwargs = {"lr": train_config["lr"], "momentum": 0.7}
        optimizer = optim.SGD(net.parameters(), **optim_kwargs)

    lr_scheduler = None
    if LR_WARMUP and config["lr_warmup_duration"] > 0:
        warmup_iters = config["lr_warmup_duration"]
        starting_factor = config["lr_warmup_starting_factor"]
        lr_scheduler_kwargs = {"start_factor": starting_factor, "end_factor": 1, "total_iters": warmup_iters}
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, **lr_scheduler_kwargs)
    else:
        warmup_iters = 0

    # Train:
    for epoch in range(1, EPOCHS + 1):
        if epoch > warmup_iters:
            lr_scheduler = None

        _, _ = train_loop(train_dataloader=train_loader, model=net, optimizer=optimizer,
                          criterion=loss, device=device,
                          first_batch=0, init_running_losses=None,
                          print_batch_interval=None,
                          checkpoint_batch_interval=None,
                          save_checkpoint_kwargs=None,
                          progress_bar=False,
                          running_acc=False)

        if lr_scheduler is not None:
            if epoch % LR_WARMUP_STEP_EPOCH_INTERVAL == 0:
                lr_scheduler.step()

        if TESTING_EPOCH_INTERVAL is not None and epoch % TESTING_EPOCH_INTERVAL == 0:
            metrics, _, roc_info = test_loop(model=net, test_dataloader=test_loader, device=device,
                                             verbose=False,
                                             progress_bar=False)
            test_acc = float(metrics['aggregate_accuracy'])
            test_macro_f1 = float(metrics['macro_f1'])
            test_obs_apnea_f1 = float(metrics['f1_by_class']['obstructive_apnea'])
            test_macro_auc = float(metrics['macro_auc'])
            if test_obs_apnea_f1 == float("nan"):
                test_obs_apnea_f1 = 0.0
            combined_score = np.mean([test_acc, test_macro_f1, test_obs_apnea_f1, test_macro_auc])

            train.report({"accuracy": test_acc,
                          "macro_f1": test_macro_f1,
                          "obstructive_apnea_f1": test_obs_apnea_f1,
                          "macro_auc": test_macro_auc,
                          "combined_score": combined_score})


if __name__ == "__main__":
    search_space = None
    initial_params = None
    if NET_TYPE == "UResIncNet":
        if OPTIMIZER == "adam":
            search_space = {
                "lr": tune.sample_from(lambda spec: 10 ** (-1 * np.random.randint(1, 4))),
                "kernel_size": tune.choice([3, 5]),
                "depth": tune.choice([4, 6, 8]),
                "layers": tune.choice([1, 2]),
                "sampling_method": tune.choice(["conv_stride", "pooling"]),
            }
            initial_params = [{
                "lr": 0.01,
                "kernel_size": 3,
                "depth": 8,
                "layers": 1,
                "sampling_method": "conv_stride"
            }]
    elif NET_TYPE == "CombinedNet":
        if OPTIMIZER == "adam":
            search_space = {
                "lr": tune.sample_from(lambda spec: 10 ** (-1 * np.random.randint(1, 4))),
                "kernel_size": tune.choice([3, 5]),
                "depth": tune.choice([4, 6, 8]),
                "layers": tune.choice([1, 2]),
                "sampling_method": tune.choice(["conv_stride", "pooling"]),
                "lstm_hidden_size": tune.sample_from(lambda spec: 2 ** (np.random.randint(4, 8))),  # 16, 32, 64, 128
                "lstm_layers": tune.choice([1, 2]),
                "lstm_dropout": tune.quniform(0.05, 0.21, 0.05),  # 0.05, 0.10, 0.15, 0.20
            }
            initial_params = [{
                "lr": 0.01,
                "kernel_size": 3,
                "depth": 8,
                "layers": 1,
                "sampling_method": "conv_stride",
                "lstm_hidden_size": 64,
                "lstm_layers": 2,
                "lstm_dropout": 0.1,
            }]

    algo = HyperOptSearch(points_to_evaluate=initial_params)
    # algo = ConcurrencyLimiter(algo, max_concurrent=4)

    scheduler = ASHAScheduler(
        max_t=5,
        grace_period=1,
        reduction_factor=4)

    tuner = tune.Tuner(
        tune.with_resources(
            full_train_loop,
            {"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="combined_score",
            mode="max",
            num_samples=15,
            scheduler=scheduler,
            search_alg=algo
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result("combined_score", "max")
    print("Best trial config: {}".format(best_result.config))

    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial final validation macro_f1: {}".format(
        best_result.metrics["macro_f1"]))
    print("Best trial final validation macro_auc: {}".format(
        best_result.metrics["macro_auc"]))

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    dfs = {result.path: result.metrics_dataframe for result in results}

    with open("tune-results.plk", mode="wb") as f:
        pickle.dump(dfs, f)

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.accuracy.plot(ax=ax, legend=False)
