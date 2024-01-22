import yaml
from pathlib import Path
import datetime, time
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchinfo import summary

from tqdm import tqdm

# Local imports:
from data_loaders_iterable import IterDataset, get_saved_train_loader, get_saved_test_loader
from pre_batched_dataloader import get_pre_batched_train_loader, get_pre_batched_test_loader, \
    get_pre_batched_test_cross_sub_loader

from UNet import UNet
from tester import test_loop

# --- START OF CONSTANTS --- #
EPOCHS = 100
BATCH_SIZE = 256
BATCH_SIZE_TEST = 32768
NUM_WORKERS = 2
LR_TO_BATCH_RATIO = 1 / 25600
LR_WARMUP = False
LR_WARMUP_EPOCH_DURATION = 3
LR_WARMUP_STEP_EPOCH_INTERVAL = 1
LR_WARMUP_STEP_BATCH_INTERVAL = 0
OPTIMIZER = "adam"  # sgd, adam
SAVE_MODEL_BATCH_INTERVAL = 1000
SAVE_MODEL_EVERY_EPOCH = True
TESTING_EPOCH_INTERVAL = 1

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
    COMPUTE_PLATFORM = config["system"]["specs"]["compute_platform"]
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1
    COMPUTE_PLATFORM = "cpu"


# --- END OF CONSTANTS --- #


def save_checkpoint(net: nn.Module, optimizer, optimizer_kwargs: dict, criterion,
                    net_type: str, identifier: str | int,
                    batch_size: int, epoch: int, batch: int = None, metrics: dict = None,
                    other_details: str = ""):
    model_path = models_path.joinpath(f"{net_type}", str(identifier), f"epoch-{epoch}")
    model_path.mkdir(parents=True, exist_ok=True)
    # identifier = 1
    # while net_path.joinpath(f"{identifier}").is_dir():
    #     identifier += 1

    txt_path = model_path.joinpath("details.txt")
    with open(txt_path, 'w') as file:
        details = [f'NET args: {net.get_args_summary()}\n',
                   f'Model: {type(net).__name__}\n',
                   f'Criterion: {type(criterion).__name__}\n',
                   f'Optimizer: {type(optimizer).__name__}\n',
                   f'Optimizer kwargs: {optimizer_kwargs}\n',
                   f'Batch size: {batch_size}\n',
                   f'Epoch: {epoch}\n',
                   f'{other_details}\n']
        file.writelines("% s\n" % line for line in details)

    state = {
        'epoch': epoch,
        'batch': batch,
        'net_class': net.__class__,
        'net_state_dict': net.state_dict(),
        'net_kwargs': net.get_kwargs(),
        'optimizer_class': optimizer.__class__,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_kwargs': optimizer_kwargs,
        'criterion_class': criterion.__class__,
        'criterion_state_dict': criterion.state_dict()
    }
    if batch is None:
        batch = "final"
    torch.save(state, model_path.joinpath(f"batch-{batch}.pt"))

    if metrics is not None:
        with open(model_path.joinpath(f"batch-{batch}-metrics.json"), 'w') as file:
            json.dump(metrics, file)


def load_checkpoint(net_type: str, identifier: str, epoch: int, batch: int):
    if batch is None:
        batch = "final"

    model_path = models_path.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}.pt")
    state = torch.load(model_path)
    net_class = state["net_class"]
    net_state = state["net_state_dict"]
    net_kwargs = state["net_kwargs"]
    optimizer_class = state["optimizer_class"]
    optimizer_state_dict = state["optimizer_state_dict"]
    optimizer_kwargs = state["optimizer_kwargs"]
    criterion_class = state["criterion_class"]
    criterion_state_dict = state["criterion_state_dict"]

    net = net_class(**net_kwargs)
    net.load_state_dict(net_state)

    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(optimizer_state_dict)

    criterion = criterion_class()
    criterion.load_state_dict(criterion_state_dict)

    return net, optimizer, criterion


def train_loop(train_dataloader: DataLoader, net: nn.Module, optimizer, criterion, lr_scheduler=None,
               device=torch.device("cpu"),
               lr_step_batch_interval: int = 10000, print_batch_interval: int = 10000,
               checkpoint_batch_interval: int = 0, save_checkpoint_kwargs: dict = None):
    if lr_step_batch_interval > 0:
        assert lr_scheduler is not None

    if checkpoint_batch_interval > 0:
        assert save_checkpoint_kwargs is not None

    # Ensure train mode:
    net.train()

    unix_time_start = time.time()

    running_loss = 0.0
    batches = len(train_dataloader)
    for (i, data) in tqdm(enumerate(train_dataloader), total=batches):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        m = nn.LogSoftmax(dim=1)
        criterion = nn.NLLLoss()
        loss = criterion(m(outputs), labels.long())
        # loss = criterion(outputs, labels.long())

        loss.backward()
        optimizer.step()

        # Update lr:
        if lr_scheduler and i % lr_step_batch_interval == lr_step_batch_interval - 1:
            lr_scheduler.step()

        # print statistics
        running_loss += loss.item()
        if print_batch_interval > 0 and i % print_batch_interval == print_batch_interval - 1:
            time_elapsed = time.time() - unix_time_start

            print(f'[Batch{i + 1:5d}/{batches:5d}]'
                  f' Running Avg loss: {running_loss / print_batch_interval:.4f}'
                  f' Secs/Batch: {time_elapsed / (i + 1) / 3600:.2f}')
            running_loss = 0.0

        # Save checkpoint:
        if checkpoint_batch_interval > 0 and i % checkpoint_batch_interval == checkpoint_batch_interval - 1:
            save_checkpoint(**save_checkpoint_kwargs, batch=i)


if __name__ == "__main__":
    models_path = PATH_TO_SUBSET1_TRAINING.joinpath("saved-models")
    models_path.mkdir(parents=True, exist_ok=True)

    if COMPUTE_PLATFORM == "opencl":
        PATH_TO_PT_OCL_DLL = Path(config["paths"]["local"]["pt_ocl_dll"])
        PATH_TO_DEPENDENCY_DLLS = Path(config["paths"]["local"]["dependency_dlls"])
        os.add_dll_directory(str(PATH_TO_DEPENDENCY_DLLS))
        torch.ops.load_library(str(PATH_TO_PT_OCL_DLL))
        device = "privateuseone:0"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Prepare train dataloader:
    # train_loader = get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = get_pre_batched_train_loader(batch_size=BATCH_SIZE, n_workers=NUM_WORKERS)
    test_loader = get_saved_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Create Network:
    net = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=5, sampling_method="conv_stride")
    net = net.to(device)

    summary(net, input_size=(BATCH_SIZE, 1, 512),
            col_names=('input_size', "output_size", "kernel_size", "num_params"), device=device)

    # Define loss:
    loss = nn.CrossEntropyLoss()

    # Set LR:
    lr = LR_TO_BATCH_RATIO * BATCH_SIZE

    # Define optimizer:
    if OPTIMIZER == "adam":
        optim_kwargs = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-08}
        optimizer = optim.Adam(net.parameters(), **optim_kwargs)
    else:  # sgd
        optim_kwargs = {"lr": lr, "momentum": 0.7}
        optimizer = optim.SGD(net.parameters(), **optim_kwargs)

    print(optim_kwargs)
    if LR_WARMUP:
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.3, end_factor=1, total_iters=3)
    else:
        lr_scheduler = None

    # Train:
    print(datetime.datetime.now())
    unix_time_start = time.time()
    checkpoint_kwargs = {"net": net,
                         "optimizer": optimizer,
                         "optimizer_kwargs": optim_kwargs,
                         "criterion": loss,
                         "net_type": "UNET",
                         "identifier": "ks5-stride-0",
                         "batch_size": BATCH_SIZE}

    for epoch in range(1, EPOCHS + 1):  # loop over the dataset multiple times
        if epoch > LR_WARMUP_EPOCH_DURATION:
            lr_scheduler = None

        checkpoint_kwargs["epoch"] = epoch
        # print(f"Batches in epoch: {batches}")
        train_loop(train_dataloader=train_loader, net=net, optimizer=optimizer, criterion=loss,
                   lr_scheduler=lr_scheduler, lr_step_batch_interval=LR_WARMUP_STEP_BATCH_INTERVAL,
                   device=device, print_batch_interval=SAVE_MODEL_BATCH_INTERVAL,
                   checkpoint_batch_interval=SAVE_MODEL_BATCH_INTERVAL, save_checkpoint_kwargs=checkpoint_kwargs)

        time_elapsed = time.time() - unix_time_start
        print(f"Epoch: {epoch} finished. Hours/Epoch: {time_elapsed / epoch / 3600}")

        if lr_scheduler and epoch % LR_WARMUP_STEP_EPOCH_INTERVAL == LR_WARMUP_STEP_EPOCH_INTERVAL - 1:
            lr_scheduler.step()

        if epoch % TESTING_EPOCH_INTERVAL == TESTING_EPOCH_INTERVAL - 1:
            metrics = test_loop(net=net, test_loader=test_loader, verbose=False)
            # Save model:
            save_checkpoint(**checkpoint_kwargs, metrics=metrics)
        elif SAVE_MODEL_EVERY_EPOCH:
            # Save model:
            save_checkpoint(**checkpoint_kwargs)

    print('Finished Training')
    print(datetime.datetime.now())

    checkpoint_kwargs["epoch"] = EPOCHS
    # Save model:
    save_checkpoint(**checkpoint_kwargs)
