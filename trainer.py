import random

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
from UResIncNet import UResIncNet
from tester import test_loop

# --- START OF CONSTANTS --- #
NET_TYPE: str = "UResIncNet"  # UNET or UResIncNet
IDENTIFIER: str = "ks3-depth8-strided-0"
LOAD_CHECKPOINT: bool = True  # True or False
LOAD_FROM_EPOCH: int | str = "last"  # epoch number or last or no
LOAD_FROM_BATCH: int | str = "last"  # batch number or last or no
EPOCHS = 100
BATCH_SIZE = 256
BATCH_SIZE_TEST = 1024
NUM_WORKERS = 2
NUM_WORKERS_TEST = 4
PRE_FETCH = 2
PRE_FETCH_TEST = 1
LR_TO_BATCH_RATIO = 1 / 25600
LR_WARMUP = False
LR_WARMUP_EPOCH_DURATION = 3
LR_WARMUP_STEP_EPOCH_INTERVAL = 1
LR_WARMUP_STEP_BATCH_INTERVAL = 0
OPTIMIZER = "adam"  # sgd, adam
SAVE_MODEL_BATCH_INTERVAL = 5000
SAVE_MODEL_EVERY_EPOCH = True
TESTING_EPOCH_INTERVAL = 101
RUNNING_LOSS_PERIOD = 100

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
    if "subset_1_saved_models_directory" in config["paths"]["local"]:
        MODELS_PATH = Path(config["paths"]["local"]["subset_1_saved_models_directory"])
    else:
        MODELS_PATH = PATH_TO_SUBSET1_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = config["system"]["specs"]["compute_platform"]
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1
    MODELS_PATH = PATH_TO_SUBSET1_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = "cpu"

MODELS_PATH.mkdir(parents=True, exist_ok=True)


# --- END OF CONSTANTS --- #


def save_checkpoint(net: nn.Module, optimizer, optimizer_kwargs: dict, criterion,
                    net_type: str, identifier: str | int,
                    batch_size: int, epoch: int, batch: int, dataloader_rng_state: torch.ByteTensor | tuple,
                    test_metrics: dict = None, running_losses: list[float] = None, other_details: str = ""):
    """
    :param net: Neural network model to save
    :param optimizer: Torch optimizer object
    :param optimizer_kwargs: Parameters used in the initialization of Optimizer
    :param criterion: The loss function object
    :param net_type: The string descriptor of the network type: UNET or UResIncNet
    :param identifier: A unique string identifier for the specific configuration of the net_type neural network
    :param batch_size: Batch size used in training
    :param epoch: Epoch of last trained batch
    :param batch: Last trained batch
    :param dataloader_rng_state: The random number generator state of dataloader's sampler.
    It is either torch.ByteTensor if it was a generator of class: torch.Generator
    or tuple if it was a generator of class: random.Random
    :param test_metrics: A dictionary containing all relevant test metrics
    :param running_losses: The running period losses of the train loop
    :param other_details: Any other details to be noted
    :return:
    """
    model_path = MODELS_PATH.joinpath(f"{net_type}", str(identifier), f"epoch-{epoch}")
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

    if running_losses is None:
        running_losses = []

    state = {
        'epoch': epoch,
        'batch': batch,
        'batch_size': batch_size,
        'net_class': net.__class__,
        'net_state_dict': net.state_dict(),
        'net_kwargs': net.get_kwargs(),
        'optimizer_class': optimizer.__class__,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_kwargs': optimizer_kwargs,
        'criterion_class': criterion.__class__,
        'criterion_state_dict': criterion.state_dict(),
        'rng_state': dataloader_rng_state,
        'running_losses': running_losses
    }

    torch.save(state, model_path.joinpath(f"batch-{batch}.pt"))

    if test_metrics is not None:
        with open(model_path.joinpath(f"batch-{batch}-test_metrics.json"), 'w') as file:
            json.dump(test_metrics, file)


def load_checkpoint(net_type: str, identifier: str, epoch: int, batch: int, device: str) \
        -> tuple[nn.Module, torch.optim.Optimizer, dict, nn.Module, torch.Generator | random.Random | None, float]:
    model_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}.pt")
    state = torch.load(model_path, map_location=device)
    if "batch_size" in state.keys():
        batch_size = int(state["batch_size"])
        if batch_size != BATCH_SIZE:
            print(f"Warning: Loading checkpoint trained with batch size: {batch_size}")

    net_class = state["net_class"]
    net_state = state["net_state_dict"]
    net_kwargs = state["net_kwargs"]
    optimizer_class = state["optimizer_class"]
    optimizer_state_dict = state["optimizer_state_dict"]
    optimizer_kwargs = state["optimizer_kwargs"]
    criterion_class = state["criterion_class"]
    criterion_state_dict = state["criterion_state_dict"]

    model = net_class(**net_kwargs)
    model.load_state_dict(net_state)
    model = model.to(device)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(optimizer_state_dict)

    criterion = criterion_class()
    criterion.load_state_dict(criterion_state_dict)

    random_number_generator = None
    if "rng_state" in state.keys():
        rng_state = state["rng_state"]
        if isinstance(rng_state, torch.ByteTensor):
            random_number_generator = torch.Generator()
            random_number_generator.set_state(rng_state)
        elif isinstance(rng_state, tuple):
            random_number_generator = random.Random()
            random_number_generator.setstate(rng_state)

    if "running_losses" in state.keys():
        run_losses = state["running_losses"]
    else:
        run_losses = None

    return model, optimizer, optimizer_kwargs, criterion, random_number_generator, run_losses


def get_saved_epochs(net_type: str, identifier: str) -> list[int]:
    model_path = MODELS_PATH.joinpath(f"{net_type}", identifier)
    if model_path.exists():
        epochs = []
        for file in model_path.iterdir():
            if file.is_dir() and "epoch-" in file.name:
                # Make sure epoch dir is not empty:
                if any(model_path.joinpath(file.name).iterdir()):
                    epoch_id = int(file.name.split('-')[1])
                    epochs.append(epoch_id)
        return sorted(epochs)
    else:
        return []


def get_saved_batches(net_type: str, identifier: str, epoch: int | str) -> list[int]:
    if epoch is None or epoch == "final" or epoch == "last":
        epoch = get_last_epoch(net_type=net_type, identifier=identifier)

    model_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}")
    if epoch != -1 and model_path.exists():
        batches = []
        for file in model_path.iterdir():
            if file.is_file() and "batch-" in file.name and ".pt" in file.name:
                name = file.name.removesuffix(".pt")
                name = name.removeprefix("batch-")
                batches.append(int(name))
        batches.sort()
        return batches
    else:
        return []


def get_last_epoch(net_type: str, identifier: str) -> int:
    epochs = get_saved_epochs(net_type=net_type, identifier=identifier)
    if len(epochs) != 0:
        last_existing_epoch = epochs[-1]
    else:
        last_existing_epoch = -1
    return last_existing_epoch


def get_last_batch(net_type: str, identifier: str, epoch: int | str) -> int:
    if epoch == -1:
        return -1

    if epoch is None or epoch == "final" or epoch == "last":
        epoch = get_last_epoch(net_type=net_type, identifier=identifier)

    batches = get_saved_batches(net_type=net_type, identifier=identifier, epoch=epoch)
    if len(batches) != 0:
        last_existing_batch = get_saved_batches(net_type=net_type, identifier=identifier, epoch=epoch)[-1]
    else:
        last_existing_batch = -1
    return last_existing_batch


def train_loop(train_dataloader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
               lr_scheduler=None, lr_step_batch_interval: int = 10000, device="cpu", first_batch: int = 0,
               initial_running_losses: list[float] = None, print_batch_interval: int = None,
               checkpoint_batch_interval: int = 0, save_checkpoint_kwargs: dict = None):
    # Resumes training from correct batch
    train_loader.sampler.first_batch_index = first_batch

    if lr_step_batch_interval > 0:
        assert lr_scheduler is not None

    if checkpoint_batch_interval > 0:
        assert save_checkpoint_kwargs is not None

    # Ensure train mode:
    model.train()
    model = model.to(device)

    unix_time_start = time.time()

    if initial_running_losses is not None:
        period_losses = initial_running_losses
    else:
        period_losses = []

    if save_checkpoint_kwargs is not None:
        epch = save_checkpoint_kwargs["epoch"]
    else:
        epch = None

    batches = len(train_dataloader)
    with tqdm(train_dataloader,
              initial=first_batch,
              total=batches,
              leave=False,
              desc="Train loop",
              unit="batch") as tqdm_dataloader:
        for (i, data) in enumerate(tqdm_dataloader, start=first_batch):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Convert to accepted dtypes: float32, float64, int64 and maybe more but not sure
            labels = labels.type(torch.int64)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Update lr:
            if lr_scheduler and i % lr_step_batch_interval == lr_step_batch_interval - 1:
                lr_scheduler.step()

            # Compute running loss:
            period_losses.append(loss.item())
            if len(period_losses) > RUNNING_LOSS_PERIOD:
                period_losses = period_losses[-RUNNING_LOSS_PERIOD:]

            running_loss = sum(period_losses) / len(period_losses)
            if epch is not None:
                tqdm_dataloader.set_postfix(running_loss=f"{running_loss:.5f}", epoch=epch)
            else:
                tqdm_dataloader.set_postfix(running_loss=f"{running_loss:.5f}")

            # print statistics
            if print_batch_interval is not None and \
                    print_batch_interval > 0 and i % print_batch_interval == print_batch_interval - 1:
                time_elapsed = time.time() - unix_time_start

                print(f'[Batch: {i + 1:5d}/{batches:5d}]'
                      f' Running Avg loss: {running_loss:.5f}'
                      f' Secs/Batch: {time_elapsed / (i + 1) / 3600:.3f}')

            # Save checkpoint:
            if checkpoint_batch_interval > 0 and i % checkpoint_batch_interval == checkpoint_batch_interval - 1:
                save_checkpoint(**save_checkpoint_kwargs, batch=i, running_losses=period_losses)


if __name__ == "__main__":
    if COMPUTE_PLATFORM == "opencl":
        PATH_TO_PT_OCL_DLL = Path(config["paths"]["local"]["pt_ocl_dll"])
        PATH_TO_DEPENDENCY_DLLS = Path(config["paths"]["local"]["dependency_dlls"])
        os.add_dll_directory(str(PATH_TO_DEPENDENCY_DLLS))
        torch.ops.load_library(str(PATH_TO_PT_OCL_DLL))
        device = "privateuseone:0"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Prepare train dataloader:
    # train_loader = get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = get_pre_batched_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PRE_FETCH)
    test_loader = get_pre_batched_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS_TEST,
                                              pre_fetch=PRE_FETCH_TEST, shuffle=False)

    # Create Network:
    last_epoch = get_last_epoch(net_type=NET_TYPE, identifier=IDENTIFIER)
    last_batch = get_last_batch(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=last_epoch)
    if LOAD_CHECKPOINT and (last_epoch == -1 or last_batch == -1):
        print("WARNING: load from checkpoint was requested but no checkpoint found! Creating new model...")
        LOAD_CHECKPOINT = False

    if LOAD_CHECKPOINT and LOAD_FROM_EPOCH != "no" and LOAD_FROM_BATCH != "no":
        if LOAD_FROM_EPOCH == "last" and LOAD_FROM_BATCH == "last":
            epoch = last_epoch
            batch = get_last_batch(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=epoch)
        elif LOAD_FROM_EPOCH == "last":
            epoch = get_last_epoch(net_type=NET_TYPE, identifier=IDENTIFIER)
            batch = LOAD_FROM_BATCH
        elif LOAD_FROM_BATCH == "last":
            epoch = LOAD_FROM_EPOCH
            batch = "final"
        else:
            epoch = LOAD_FROM_EPOCH
            batch = LOAD_FROM_BATCH

        net, optimizer, optim_kwargs, loss, rng, initial_running_losses = load_checkpoint(net_type=NET_TYPE,
                                                                                          identifier=IDENTIFIER,
                                                                                          epoch=epoch,
                                                                                          batch=batch,
                                                                                          device=device)

        if rng is not None:
            # Save dataloader rng state in order to be able to resume training from the same batch
            if isinstance(rng, train_loader.sampler.rng.__class__):
                train_loader.sampler.rng = rng
            else:
                print(f"WARNING: saved rng class {rng.__class__} is different from dataloader class: "
                      f"{train_loader.sampler.rng.__class__} ! Correct batch order cannot be determined. "
                      f"Training may result in duplicate batch training!")
        else:
            print("WARNING: Could not load rng state. Correct batch order cannot be determined. "
                  f"Training may result in duplicate batch training!")

        # Check if it is final batch
        if batch == len(train_loader) - 1:
            start_from_epoch = epoch + 1
            start_from_batch = 0
        else:
            start_from_epoch = epoch
            start_from_batch = batch + 1
    else:
        if NET_TYPE == "UNET":
            net = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=5,
                       sampling_method="conv_stride")
        else:
            net = UResIncNet(nclass=5, in_chans=1, max_channels=512, depth=8, kernel_size=3, sampling_factor=2,
                             sampling_method="conv_stride", skip_connection=True)

        initial_running_losses = None
        start_from_epoch = 1
        start_from_batch = 0

        # Define loss:
        loss = nn.CrossEntropyLoss()

        # Set LR:
        lr = LR_TO_BATCH_RATIO * BATCH_SIZE

        # Model should go to device first before initializing optimizer:
        net = net.to(device)

        # Define optimizer:
        if OPTIMIZER == "adam":
            optim_kwargs = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-08}
            optimizer = optim.Adam(net.parameters(), **optim_kwargs)
        else:  # sgd
            optim_kwargs = {"lr": lr, "momentum": 0.7}
            optimizer = optim.SGD(net.parameters(), **optim_kwargs)

    summary(net, input_size=(BATCH_SIZE, 1, 512),
            col_names=('input_size', "output_size", "kernel_size", "num_params"), device=device)

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
                         "net_type": NET_TYPE,
                         "identifier": IDENTIFIER,
                         "batch_size": BATCH_SIZE}

    batches_in_epoch = len(train_loader)
    # loop over the dataset multiple times:
    for epoch in tqdm(range(start_from_epoch, EPOCHS + 1), initial=start_from_epoch - 1, total=EPOCHS,
                      desc="Epochs finished", leave=True):

        if epoch > LR_WARMUP_EPOCH_DURATION:
            lr_scheduler = None

        checkpoint_kwargs["epoch"] = epoch
        if isinstance(train_loader.sampler.rng, torch.Generator):
            checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.get_state()
        else:
            checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.getstate()

        # print(f"Batches in epoch: {batches}")
        train_loop(train_dataloader=train_loader, model=net, optimizer=optimizer, criterion=loss,
                   lr_scheduler=lr_scheduler, lr_step_batch_interval=LR_WARMUP_STEP_BATCH_INTERVAL,
                   device=device, first_batch=start_from_batch, initial_running_losses=initial_running_losses,
                   print_batch_interval=None, checkpoint_batch_interval=SAVE_MODEL_BATCH_INTERVAL,
                   save_checkpoint_kwargs=checkpoint_kwargs)

        time_elapsed = time.time() - unix_time_start
        # print(f"Epoch: {epoch} finished. Hours/Epoch: {time_elapsed / epoch / 3600}")

        if lr_scheduler and epoch % LR_WARMUP_STEP_EPOCH_INTERVAL == LR_WARMUP_STEP_EPOCH_INTERVAL - 1:
            lr_scheduler.step()

        if epoch % TESTING_EPOCH_INTERVAL == 0:
            # print(f"Testing epoch: {epoch}")
            metrics = test_loop(model=net, test_dataloader=test_loader, device=device, verbose=False, progress_bar=True)
            print(f"Test accuracy: {metrics['aggregate_accuracy']}")
            # Save model:
            save_checkpoint(**checkpoint_kwargs, batch=batches_in_epoch - 1, test_metrics=metrics)
        elif SAVE_MODEL_EVERY_EPOCH:
            # Save model:
            save_checkpoint(**checkpoint_kwargs, batch=batches_in_epoch - 1)

        start_from_batch = 0
    print('Finished Training')
    print(datetime.datetime.now())

    checkpoint_kwargs["epoch"] = EPOCHS

    # Save model:
    save_checkpoint(**checkpoint_kwargs, batch=batches_in_epoch - 1)
