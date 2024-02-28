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

from UNet import UNet, ConvNet
from UResIncNet import UResIncNet, ResIncNet
from tester import test_loop, save_metrics, save_confusion_matrix, save_rocs

# --- START OF CONSTANTS --- #

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
CLASSIFY_PER_SAMPLE = True
LR_TO_BATCH_RATIO = 1 / 25600
LR_WARMUP = True
LR_WARMUP_DURATION = 5
LR_WARMUP_STEP_EPOCH_INTERVAL = 1
OPTIMIZER = "adam"  # sgd, adam
SAVE_MODEL_BATCH_INTERVAL = 999999999
SAVE_MODEL_EVERY_EPOCH = True
TESTING_EPOCH_INTERVAL = 1
RUNNING_LOSS_PERIOD = 100

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
    KERNEL_SIZE = int(config["variables"]["models"]["kernel_size"])
    DEPTH = int(config["variables"]["models"]["depth"])
    LAYERS = int(config["variables"]["models"]["layers"])
    SAMPLING_METHOD = config["variables"]["models"]["sampling_method"]
    use_weighted_loss = config["variables"]["models"]["use_weighted_loss"]
    if use_weighted_loss:
        # Class weights:
        # Subset1: Balance based on samples: old[1, 176, 23, 12, 3] now: [1, 95, 13, 7, 2]
        # Subset1 UResIncNET-"ks3-depth8-strided-0": Balance based on final recall: [1, 59, 20, 17, 1]
        assert "class_weights" in config["variables"]["models"]
        cw_tmp = config["variables"]["models"]["class_weights"]
        assert cw_tmp is not None
        CLASS_WEIGHTS = cw_tmp
    else:
        CLASS_WEIGHTS = None
else:
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_TRAINING = PATH_TO_SUBSET
    MODELS_PATH = PATH_TO_SUBSET_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = "cpu"
    NET_TYPE: str = "UResIncNet"  # UNET or UResIncNet
    IDENTIFIER: str = "ks3-depth8-strided-0"  # ks5-depth5-layers2-strided-0 or ks3-depth8-strided-0
    KERNEL_SIZE = 3
    DEPTH = 8
    LAYERS = 1
    SAMPLING_METHOD = "conv_stride"
    use_weighted_loss = False
    CLASS_WEIGHTS = None
MODELS_PATH.mkdir(parents=True, exist_ok=True)


# --- END OF CONSTANTS --- #


def save_checkpoint(net_type: str, identifier: str | int, epoch: int, batch: int, batch_size: int, net: nn.Module, net_kwargs: dict,
                    optimizer, optimizer_kwargs: dict | None, criterion, criterion_kwargs: dict | None,
                    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None, lr_scheduler_kwargs: dict | None,
                    dataloader_rng_state: torch.ByteTensor | tuple, running_losses: list[float] = None,
                    test_metrics: dict = None, test_cm: list[list[float]] = None,
                    roc_info: dict[str: dict[str: list | float]] = None,
                    other_details: str = ""):
    """
    :param roc_info:
    :param lr_scheduler_kwargs:
    :param lr_scheduler:
    :param criterion_kwargs:
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
    :param test_cm: Confusion matrix
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
        details = [f'NET kwargs: {net_kwargs}\n',
                   f'Model: {type(net).__name__}\n',
                   f'Criterion: {type(criterion).__name__}\n',
                   f'Criterion kwargs: {criterion_kwargs}\n',
                   f'Optimizer: {type(optimizer).__name__}\n',
                   f'Optimizer kwargs: {optimizer_kwargs}\n',
                   f'Batch size: {batch_size}\n',
                   f'Epoch: {epoch}\n',
                   f'{other_details}\n']
        file.writelines("% s\n" % line for line in details)

    state = {
        'epoch': epoch,
        'batch': batch,
        'batch_size': batch_size,
        'net_class': net.__class__,
        'net_state_dict': net.state_dict(),
        'net_kwargs': net_kwargs,
        'optimizer_class': optimizer.__class__,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_kwargs': optimizer_kwargs,
        'criterion_class': criterion.__class__,
        'criterion_state_dict': criterion.state_dict(),
        'criterion_kwargs': criterion_kwargs,
        'rng_state': dataloader_rng_state,
        'running_losses': running_losses
    }
    if lr_scheduler is not None and lr_scheduler_kwargs is not None:
        state["lr_scheduler_class"] = lr_scheduler.__class__
        state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    torch.save(state, model_path.joinpath(f"batch-{batch}.pt"))

    if test_metrics is not None:
        save_metrics(metrics=test_metrics, net_type=net_type, identifier=identifier, epoch=epoch, batch=batch)
        # with open(model_path.joinpath(f"batch-{batch}-test_metrics.json"), 'w') as file:
        #     json.dump(test_metrics, file)
    if test_cm is not None:
        save_confusion_matrix(test_cm, net_type=net_type, identifier=identifier, epoch=epoch, batch=batch)

    if roc_info is not None:
        save_rocs(roc_info, net_type=net_type, identifier=identifier, epoch=epoch, batch=batch, save_plot=True)


def load_checkpoint(net_type: str, identifier: str, epoch: int, batch: int, device: str) \
        -> tuple[
            nn.Module, dict, torch.optim.Optimizer, dict, nn.Module, dict, torch.optim.lr_scheduler.LRScheduler, dict,
            torch.Generator | random.Random | None, float]:
    model_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}.pt")
    state = torch.load(model_path)
    if "batch_size" in state.keys():
        batch_size = int(state["batch_size"])
        if batch_size != BATCH_SIZE:
            print(f"Warning: Loading checkpoint trained with batch size: {batch_size}")

    net_class = state["net_class"]
    net_state = state["net_state_dict"]
    model_kwargs = state["net_kwargs"]
    optimizer_class = state["optimizer_class"]
    optimizer_state_dict = state["optimizer_state_dict"]
    criterion_class = state["criterion_class"]
    criterion_state_dict = state["criterion_state_dict"]

    model = net_class(**model_kwargs)
    model.load_state_dict(net_state)
    model = model.to(device)

    # Initialize Optimizer:
    if "optimizer_kwargs" in state.keys() and state["optimizer_kwargs"] is not None:
        optimizer_kwargs = state["optimizer_kwargs"]
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    else:
        optimizer_kwargs = None
        optimizer = optimizer_class(model.parameters())

    optimizer.load_state_dict(optimizer_state_dict)

    # Initialize Loss:
    if "criterion_kwargs" in state.keys() and state["criterion_kwargs"] is not None:
        criterion_kwargs = state["criterion_kwargs"]
        for k, v in criterion_kwargs.items():
            if isinstance(v, torch.Tensor):
                criterion_kwargs[k] = v.to(device)
        criterion = criterion_class(**criterion_kwargs)
    else:
        criterion_kwargs = None
        criterion = criterion_class()
    criterion.load_state_dict(criterion_state_dict)

    lr_scheduler = None
    lr_scheduler_kwargs = None
    if "lr_scheduler_class" in state.keys() and state["lr_scheduler_class"] is not None:
        if "lr_scheduler_kwargs" in state.keys() and state["lr_scheduler_kwargs"] is not None:
            lr_scheduler_class = state["lr_scheduler_class"]
            lr_scheduler_kwargs = state["lr_scheduler_kwargs"]
            lr_scheduler = lr_scheduler_class(optimizer=optimizer, **lr_scheduler_kwargs)

    # Resume RNG state:
    random_number_generator = None
    if "rng_state" in state.keys():
        rng_state = state["rng_state"]
        if isinstance(rng_state, (torch.Tensor, torch.ByteTensor)):
            rng_state = rng_state.type(torch.ByteTensor)
            random_number_generator = torch.Generator()
            random_number_generator.set_state(rng_state)
        elif isinstance(rng_state, tuple):
            random_number_generator = random.Random()
            random_number_generator.setstate(rng_state)

    if "running_losses" in state.keys():
        run_losses = state["running_losses"]
    else:
        run_losses = None

    return (model, model_kwargs, optimizer, optimizer_kwargs, criterion, criterion_kwargs,
            lr_scheduler, lr_scheduler_kwargs, random_number_generator, run_losses)


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


def get_lr(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_window_label(window_labels: torch.tensor) -> tuple[torch.tensor, float]:
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


def train_loop(train_dataloader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
               device="cpu", first_batch: int = 0, init_running_losses: list[float] = None,
               print_batch_interval: int = None, checkpoint_batch_interval: int = None,
               save_checkpoint_kwargs: dict = None):
    # Resumes training from correct batch
    train_loader.sampler.first_batch_index = first_batch

    if checkpoint_batch_interval > 0:
        assert save_checkpoint_kwargs is not None

    # Ensure train mode:
    model.train()
    model = model.to(device)

    unix_time_start = time.time()

    if init_running_losses is not None:
        period_losses = init_running_losses
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

            if outputs.shape.numel() != inputs.shape.numel() * 5 and labels.shape[0] != labels.shape.numel():
                # Per window classification nut labels per sample:
                labels_by_window = torch.zeros((labels.shape[0]), device=device, dtype=torch.int64)
                for batch_index in range(labels.shape[0]):
                    labels_by_window[batch_index] = get_window_label(labels[batch_index, :])[0].to(device)
                batch_loss = criterion(outputs, labels_by_window)
            else:
                batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            # Compute running loss:
            period_losses.append(batch_loss.item())

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
            if checkpoint_batch_interval is not None and i % checkpoint_batch_interval == checkpoint_batch_interval - 1:
                save_checkpoint(batch=i, running_losses=period_losses, **save_checkpoint_kwargs)


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

    # Find out window size:
    sample_batch = next(iter(train_loader))
    window_size = sample_batch.shape[2]
    print(f"Window size: {window_size}. Batch size: {BATCH_SIZE}")

    # Create Network:
    last_epoch = get_last_epoch(net_type=NET_TYPE, identifier=IDENTIFIER)
    last_batch = get_last_batch(net_type=NET_TYPE, identifier=IDENTIFIER, epoch=last_epoch)
    if LOAD_CHECKPOINT and (last_epoch == -1 or last_batch == -1):
        print("WARNING: load from checkpoint was requested but no checkpoint found! Creating new model...")
        LOAD_CHECKPOINT = False

    # Class weights:
    weights = None
    if CLASS_WEIGHTS is not None:
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

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

        net, net_kwargs, optimizer, optim_kwargs, loss, loss_kwargs, \
            lr_scheduler, lr_scheduler_kwargs, rng, initial_running_losses = load_checkpoint(
            net_type=NET_TYPE,
            identifier=IDENTIFIER,
            epoch=epoch,
            batch=batch,
            device=device)
        # loss.weight = weights

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
        net = None
        net_kwargs = {}
        if NET_TYPE == "UNET":
            net = UNet(nclass=5, in_chans=1, max_channels=512, depth=DEPTH, layers=LAYERS, kernel_size=KERNEL_SIZE,
                       sampling_method=SAMPLING_METHOD)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "UResIncNet":
            net = UResIncNet(nclass=5, in_chans=1, max_channels=512, depth=DEPTH, layers=LAYERS,
                             kernel_size=KERNEL_SIZE,
                             sampling_factor=2, sampling_method=SAMPLING_METHOD, skip_connection=True)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "ConvNet":
            net = ConvNet(nclass=5, in_size=window_size, in_chans=1, max_channels=512, depth=DEPTH, layers=LAYERS,
                          kernel_size=KERNEL_SIZE, sampling_method=SAMPLING_METHOD)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "ResIncNet":
            net = ResIncNet(nclass=5, in_size=window_size, in_chans=1, max_channels=512, depth=DEPTH, layers=LAYERS,
                            kernel_size=KERNEL_SIZE,
                            sampling_factor=2, sampling_method=SAMPLING_METHOD, skip_connection=True)
            net_kwargs = net.get_kwargs()

        initial_running_losses = None
        lr_scheduler = None
        lr_scheduler_kwargs = None
        start_from_epoch = 1
        start_from_batch = 0

        # Define loss:
        if weights is not None:
            loss_kwargs = {"weight": weights}
            loss = nn.CrossEntropyLoss(**loss_kwargs)
        else:
            loss_kwargs = None
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

    # torchinfo summary
    print(NET_TYPE)
    print(IDENTIFIER)
    summary(net, input_size=(BATCH_SIZE, 1, 512),
            col_names=('input_size', "output_size", "kernel_size", "num_params"), device=device)

    print(optim_kwargs)

    if lr_scheduler is None:
        last_completed_epoch = start_from_epoch - 1
        if LR_WARMUP and last_completed_epoch < LR_WARMUP_DURATION:
            warmup_iters = LR_WARMUP_DURATION - last_completed_epoch
            starting_factor = 0.25 + last_completed_epoch * (1 - 0.25) / LR_WARMUP_DURATION
            lr_scheduler_kwargs = {"start_factor": starting_factor, "end_factor": 1, "total_iters": warmup_iters}
            lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.3, end_factor=1,
                                                       total_iters=warmup_iters)

    # Train:
    print(datetime.datetime.now())
    unix_time_start = time.time()
    checkpoint_kwargs = {"net": net,
                         "net_kwargs": net_kwargs,
                         "optimizer": optimizer,
                         "optimizer_kwargs": optim_kwargs,
                         "criterion": loss,
                         "criterion_kwargs": loss_kwargs,
                         "net_type": NET_TYPE,
                         "lr_scheduler": lr_scheduler,
                         "lr_scheduler_kwargs": lr_scheduler_kwargs,
                         "identifier": IDENTIFIER,
                         "batch_size": BATCH_SIZE}
    batches_in_epoch = len(train_loader)
    # loop over the dataset multiple times:
    with tqdm(range(start_from_epoch, EPOCHS + 1), initial=start_from_epoch - 1, total=EPOCHS,
              desc="Epochs finished", leave=True) as tqdm_epochs:
        for epoch in tqdm_epochs:

            if epoch > LR_WARMUP_DURATION:
                lr_scheduler = None

            checkpoint_kwargs["epoch"] = epoch
            if isinstance(train_loader.sampler.rng, torch.Generator):
                checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.get_state()
            else:
                checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.getstate()

            # print(f"Batches in epoch: {batches}")
            train_loop(train_dataloader=train_loader, model=net, optimizer=optimizer, criterion=loss, device=device,
                       first_batch=start_from_batch, init_running_losses=initial_running_losses,
                       print_batch_interval=None, checkpoint_batch_interval=SAVE_MODEL_BATCH_INTERVAL,
                       save_checkpoint_kwargs=checkpoint_kwargs)

            time_elapsed = time.time() - unix_time_start
            # print(f"Epoch: {epoch} finished. Hours/Epoch: {time_elapsed / epoch / 3600}")

            if lr_scheduler is not None:
                if epoch % LR_WARMUP_STEP_EPOCH_INTERVAL == 0:
                    lr_scheduler.step()
                tqdm_epochs.set_postfix(current_base_lr=f"{lr_scheduler.get_last_lr()[0]:.5f}")

            if TESTING_EPOCH_INTERVAL is not None and epoch % TESTING_EPOCH_INTERVAL == 0:
                # print(f"Testing epoch: {epoch}")
                metrics, cm, roc_info = test_loop(model=net, test_dataloader=test_loader, device=device, verbose=False,
                                                  progress_bar=True)
                test_acc = metrics['aggregate_accuracy']
                tqdm_epochs.set_postfix(epoch_test_acc=f"{test_acc:.5f}")
                # Save model:
                save_checkpoint(batch=batches_in_epoch - 1, test_metrics=metrics, test_cm=cm, roc_info=roc_info,
                                **checkpoint_kwargs)
            elif SAVE_MODEL_EVERY_EPOCH:
                # Save model:
                save_checkpoint(batch=batches_in_epoch - 1, **checkpoint_kwargs)

            start_from_batch = 0
            initial_running_losses = None
    print('Finished Training')
    print(datetime.datetime.now())

    checkpoint_kwargs["epoch"] = EPOCHS
    if isinstance(train_loader.sampler.rng, torch.Generator):
        checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.get_state()
    else:
        checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.getstate()

    # Save model:
    save_checkpoint(batch=batches_in_epoch - 1, **checkpoint_kwargs)
