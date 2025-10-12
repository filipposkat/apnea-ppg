import random
from typing import Tuple, Any

import monai.losses
import yaml
from pathlib import Path
import datetime, time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader, IterableDataset
from torchinfo import summary
from tqdm import tqdm

# Local imports:
from data_loaders_iterable import IterDataset, get_saved_train_loader, get_saved_test_loader
from pre_batched_dataloader import get_pre_batched_train_loader, get_pre_batched_test_loader, \
    get_pre_batched_test_cross_sub_loader, get_available_batch_sizes

from common import detect_desaturations_profusion
from UNet import UNet, ConvNet
from UResIncNet import UResIncNet, ResIncNet
from CombinedNet import CombinedNet
from tester import test_loop, save_metrics, save_confusion_matrix, save_rocs, save_prcs

# --- START OF CONSTANTS --- #

LOAD_CHECKPOINT: bool = True  # True or False
LOAD_FROM_EPOCH: int | str = "last"  # epoch number or last or no
LOAD_FROM_BATCH: int | str = "last"  # batch number or last or no
EPOCHS = 20
BATCH_SIZE = "auto"  # 256
BATCH_SIZE_TEST = "auto"  # 1024
EARLY_STOPPING = True  # When loading from checkpoint, early stopping resets
EARLY_STOPPING_METRIC = "aggregate_mcc"  # It should be one of the metrics computed by tester.test_loop()
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_DELTA = 0.01
NUM_WORKERS = 2
NUM_WORKERS_TEST = 4
PRE_FETCH = 2
PRE_FETCH_TEST = 1
CLASSIFY_PER_SAMPLE = True
FIRST_OUT_CHANS = 4
LR_TO_BATCH_RATIO = 1 / 25600  # If lr is defined in config then this will be omitted
LR_WARMUP = False
LR_WARMUP_ASCENDING = True
LR_WARMUP_DURATION = 3
LR_WARMUP_STEP_EPOCH_INTERVAL = 1
CEL_FL_WEIGHT = 0.7  # Weight of the CEL or FL in the combined losses (cce_dl, cce_gdl, cce_gwdl, fl_gdl)
LEARNABLE_CEL_WEIGHT = False  # Supported only for cel_gdl and cel_gwdl
OPTIMIZER = "adam"  # sgd, adam
SAVE_MODEL_BATCH_INTERVAL = 999999999
SAVE_MODEL_EVERY_EPOCH = True
TESTING_EPOCH_INTERVAL = 1
RUNNING_LOSS_PERIOD = 100

if __name__ == "__main__":
    if BATCH_SIZE == "auto" or BATCH_SIZE_TEST == "auto":
        available_train_bs, available_test_bs, _ = get_available_batch_sizes()
        if BATCH_SIZE == "auto":
            BATCH_SIZE = min([bs for bs in available_train_bs])
        if BATCH_SIZE_TEST == "auto":
            BATCH_SIZE_TEST = max([bs for bs in available_test_bs])

    LR = LR_TO_BATCH_RATIO * BATCH_SIZE

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

    if "loss" in config["variables"]:
        LOSS_FUNCTION = config["variables"]["loss"]["loss_func"]
        use_weighted_loss = config["variables"]["loss"]["use_weighted_loss"]
    else:
        LOSS_FUNCTION = "cel"
        use_weighted_loss = False

    if "lr" in config["variables"]["optimizer"]:
        LR = float(config["variables"]["optimizer"]["lr"])
        if "type" in config["variables"]["optimizer"]:
            OPTIMIZER = str(config["variables"]["optimizer"]["type"])
        if "warmup" in config["variables"]["optimizer"]:
            LR_WARMUP = config["variables"]["optimizer"]["warmup"]
        if "warmup_ascending" in config["variables"]["optimizer"]:
            LR_WARMUP_ASCENDING = config["variables"]["optimizer"]["warmup_ascending"]
    KERNEL_SIZE = int(config["variables"]["models"]["kernel_size"])
    DEPTH = int(config["variables"]["models"]["depth"])
    LAYERS = int(config["variables"]["models"]["layers"])
    SAMPLING_METHOD = config["variables"]["models"]["sampling_method"]
    DROPOUT = float(config["variables"]["models"]["dropout"])
    if "first_out_chans" in config["variables"]["models"]:
        FIRST_OUT_CHANS = int(config["variables"]["models"]["first_out_chans"])
        assert FIRST_OUT_CHANS % 4 == 0
    if "lstm_max_features" in config["variables"]["models"]:
        LSTM_MAX_FEATURES = int(config["variables"]["models"]["lstm_max_features"])
        LSTM_LAYERS = int(config["variables"]["models"]["lstm_layers"])
        LSTM_DROPOUT = float(config["variables"]["models"]["lstm_dropout"])
    else:
        LSTM_MAX_FEATURES = 128
        LSTM_LAYERS = 2
        LSTM_DROPOUT = 0.1

    if use_weighted_loss:
        # Class weights:
        # Subset1: Balance based on samples: old[1, 176, 23, 12, 3] now: [1, 95, 13, 7, 2]
        # Subset1 UResIncNET-"ks3-depth8-strided-0": Balance based on final recall: [1, 59, 20, 17, 1]
        assert "class_weights" in config["variables"]["loss"]
        cw_tmp = config["variables"]["loss"]["class_weights"]
        assert cw_tmp is not None
        if CONVERT_SPO2DESAT_TO_NORMAL:
            assert len(cw_tmp) == 4
        CLASS_WEIGHTS = cw_tmp
    else:
        CLASS_WEIGHTS = None

    CUSTOM_WEIGHT_INIT = False
    if "custom_net_weight_init" in config["variables"]["models"]:
        if config["variables"]["models"]["custom_net_weight_init"]:
            CUSTOM_WEIGHT_INIT = True
    if "neg_slope" in config["variables"]["models"]:
        NEG_SLOPE = float(config["variables"]["models"]["neg_slope"])
    else:
        NEG_SLOPE = 0.2

    if "convert_spo2_to_dst_labels" in config["variables"]["dataset"]:
        CONVERT_SPO2_TO_DST_LABELS = config["variables"]["dataset"]["convert_spo2_to_dst_labels"]
    else:
        CONVERT_SPO2_TO_DST_LABELS = False
else:
    subset_id = 1
    CONVERT_SPO2DESAT_TO_NORMAL = False
    N_INPUT_CHANNELS = 1
    PATH_TO_SUBSET = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET_TRAINING = PATH_TO_SUBSET
    MODELS_PATH = PATH_TO_SUBSET_TRAINING.joinpath("saved-models")
    COMPUTE_PLATFORM = "cpu"
    NET_TYPE: str = "UResIncNet"  # UNET, UResIncNet
    IDENTIFIER: str = "ks3-depth8-strided-0"  # ks5-depth5-layers2-strided-0 or ks3-depth8-strided-0
    KERNEL_SIZE = 3
    DEPTH = 8
    LAYERS = 1
    SAMPLING_METHOD = "conv_stride"
    DROPOUT = 0.0
    NEG_SLOPE = 0.2
    LSTM_MAX_FEATURES = 128
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.1
    LOSS_FUNCTION = "cel"
    use_weighted_loss = False
    CLASS_WEIGHTS = None
    CUSTOM_WEIGHT_INIT = False
MODELS_PATH.mkdir(parents=True, exist_ok=True)

if LOSS_FUNCTION != "cel":
    import monai

    if "dl" in LOSS_FUNCTION:
        # Dice Loss family
        # from dice_loss import DiceLoss
        # from kornia.losses import DiceLoss
        from monai.losses import DiceLoss
        from custom_losses import CelDlLoss, FlDlLoss, FlGdlLoss

        if "gdl" in LOSS_FUNCTION:
            from monai.losses import GeneralizedDiceLoss
            from custom_losses import CelGdlLoss, FlGdlLoss

            if use_weighted_loss:
                print("Loss function GDL uses inherently weights which are calculated automatically. Ignoring given "
                      "class"
                      "weights!")
            else:
                print("WARNING: Loss function GDL uses inherently weights which are calculated automatically.")
        elif "gwdl" in LOSS_FUNCTION:
            # pip install git+https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss.git
            # from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss
            from monai.losses import GeneralizedWassersteinDiceLoss
            from custom_losses import CelGwdlLoss

            print("WARNING: Loss function GDL uses inherently weights which are calculated automatically.")
            if use_weighted_loss and LOSS_FUNCTION == "gwdl":
                print("'use_weighted_loss': True, changes weighting scheme to the same used in GDL "
                      "(inversly proportional to class size)!")
            else:
                print(" Using default weighting scheme.")
    elif "fl" in LOSS_FUNCTION:
        from monai.losses import FocalLoss
        from custom_losses import FlGdlLoss
        # from kornia.losses import FocalLoss


# --- END OF CONSTANTS --- #

class EarlyStopping:
    def __init__(self, patience=5, delta=0, maximize=True, verbose=False):
        self.patience = patience
        self.delta = delta
        self.maximize = maximize
        self.verbose = verbose
        self.best_metric = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_metric):
        if self.best_metric is None or (self.maximize and (val_metric > self.best_metric + self.delta)) or (
                not self.maximize and (val_metric < self.best_metric - self.delta)):
            self.best_metric = val_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


def save_checkpoint(net_type: str, identifier: str | int, epoch: int, batch: int, batch_size: int,
                    net: nn.Module, net_kwargs: dict,
                    optimizer, optimizer_kwargs: dict | None,
                    criterion, criterion_kwargs: dict | None,
                    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None, lr_scheduler_kwargs: dict | None,
                    dataloader_rng_state: torch.ByteTensor | tuple,
                    running_losses: list[float] = None, running_loss: float = None, running_accuracy: float = None,
                    test_metrics: dict = None, test_cm: list[list[float]] = None,
                    roc_info: dict[str: dict[str: list | float]] = None,
                    pr_info: dict[str: dict[str: list | float]] = None,
                    device: str = "infer",
                    other_details: str = ""):
    """
    :param net: Neural network model to save
    :param net_kwargs: Parameters used in the initialization of net
    :param optimizer: Torch optimizer object
    :param optimizer_kwargs: Parameters used in the initialization of Optimizer
    :param criterion: The loss function object
    :param criterion_kwargs: Parameters used in the initialization of criterion
    :param lr_scheduler: Torch LR scheduler object
    :param lr_scheduler_kwargs: Parameters used in the initialization of lr_scheduler
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
    :param roc_info: dict with the test ROC curve data
    :param pr_info: dict with the test PR curve data
    :param running_losses: The running period losses of the train loop
    :param running_loss: The last running loss in the epoch
    :param running_accuracy: The last running accuracy in the epoch
    :param device: if "infer", tensors are kept to their current devices, otherwise all tensors are moved to the
    specified device
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
                   f'LR scheduler: {type(lr_scheduler).__name__}\n',
                   f'LR scheduler kwargs: {lr_scheduler_kwargs}\n',
                   f'Batch size: {batch_size}\n',
                   f'Epoch: {epoch}\n',
                   f'{other_details}\n']
        file.writelines("% s\n" % line for line in details)

    if device != "infer":
        net = net.to(device)
        optimizer = optimizer.to(device)
        criterion = criterion.to(device)
        if not isinstance(dataloader_rng_state, torch.ByteTensor):
            dataloader_rng_state = dataloader_rng_state.to(device)

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

    if pr_info is not None:
        save_prcs(pr_info, net_type=net_type, identifier=identifier, epoch=epoch, batch=batch, save_plot=True)

    if running_losses is not None:
        running_loss = sum(running_losses) / len(running_losses)

    if running_loss is not None:
        train_metrics_dict = {"train_running_loss": running_loss}
        if running_accuracy is not None:
            train_metrics_dict["train_running_accuracy"] = running_accuracy

        metrics_path = model_path.joinpath(f"batch-{batch}-train_metrics.json")
        with open(metrics_path, 'w') as file:
            json.dump(train_metrics_dict, file)


def load_checkpoint(net_type: str, identifier: str, epoch: int, batch: int, device: str) \
        -> tuple[
            nn.Module, dict, torch.optim.Optimizer, dict, nn.Module, dict, torch.optim.lr_scheduler.LRScheduler, dict,
            torch.Generator | random.Random | None, float]:
    model_path = MODELS_PATH.joinpath(f"{net_type}", identifier, f"epoch-{epoch}", f"batch-{batch}.pt")

    if isinstance(device, str) and "ocl" in device:
        state = torch.load(model_path, map_location={"cpu": "cpu", "cuda:0": "cpu"}, weights_only=False)
    else:
        state = torch.load(model_path, map_location={"cpu": "cpu", "cuda:0": str(device)}, weights_only=False)

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

    if "lstm_max_channels" in model_kwargs:
        model_kwargs.pop("lstm_max_channels")

    model = net_class(**model_kwargs)
    model.load_state_dict(net_state)
    model = model.to(device)

    # Initialize Loss:
    if "criterion_kwargs" in state.keys() and state["criterion_kwargs"] is not None:
        criterion_kwargs = state["criterion_kwargs"]
        for k, v in criterion_kwargs.items():
            if isinstance(v, torch.Tensor):
                criterion_kwargs[k] = v.to(device)
        criterion = criterion_class(**criterion_kwargs)
    else:
        print("WARNING: No criterion kwargs found in checkpoint. Using default criterion init values.")
        criterion_kwargs = None
        criterion = criterion_class()

    if isinstance(criterion, nn.CrossEntropyLoss):
        criterion.load_state_dict(criterion_state_dict)
    else:
        try:
            criterion.load_state_dict(criterion_state_dict, strict=False)
        except RuntimeError:
            # print("INFO: Failed to load state_dict to checkpoint criterion. Skipping state loading.")
            pass

    # Initialize Optimizer:
    # Define optimizer:
    if len(list(criterion.parameters())) > 0:
        optim_params = list(model.parameters()) + list(criterion.parameters())
    else:
        optim_params = model.parameters()

    if "optimizer_kwargs" in state.keys() and state["optimizer_kwargs"] is not None:
        optimizer_kwargs = state["optimizer_kwargs"]
        optimizer = optimizer_class(optim_params, **optimizer_kwargs)
    else:
        optimizer_kwargs = None
        optimizer = optimizer_class(optim_params)

    optimizer.load_state_dict(optimizer_state_dict)

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
    model_path = MODELS_PATH.joinpath(f"{net_type}", str(identifier))
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

    model_path = MODELS_PATH.joinpath(str(net_type), str(identifier), f"epoch-{epoch}")
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


def detect_desaturations_profusion_torch(
        spo2_values: torch.Tensor,
        sampling_rate: float,
        min_drop: float = 3,
        max_fall_rate: float = 4,
        max_plateau: float = 60,
        max_drop_threshold: float = 50,
        min_drop_duration: float = 1,
        max_drop_duration: float = None
):
    """
    Tensor-based version of detect_desaturations_profusion().
    Works with torch tensors on any device (CPU or CUDA).
    """

    device = spo2_values.device

    dst_lbls = torch.zeros(spo2_values.shape.numel(), dtype=torch.uint8, device=device)

    desaturation_events = 0
    min_drop_samples = int(min_drop_duration * sampling_rate)
    max_drop_samples = int(max_drop_duration * sampling_rate) if max_drop_duration else spo2_values.shape[0]
    max_plateau_samples = int(max_plateau * sampling_rate)

    i = 0
    length = spo2_values.shape[0]

    # Iterative loop (cannot be vectorized trivially because of local min/max search)
    while i < length - 1:
        # Look for a local zenith
        while i < length - 1 and (spo2_values[i + 1] >= spo2_values[i] or spo2_values[i] > 100):
            i += 1
        zenith = spo2_values[i].item()

        # Start looking for a desaturation event
        start = i
        while i < length - 1 and (spo2_values[i + 1] <= spo2_values[i] or spo2_values[i] < 50):
            i += 1

        nadir = spo2_values[i].item()
        end = i

        for j in range(i + 1, i + max_plateau_samples):
            if j >= length:
                break
            val = spo2_values[j].item()
            if 100 > val > zenith:
                break
            elif nadir >= val > 50:
                nadir = val
                i = j
                end = j
            elif (zenith - min_drop) >= val > 50:
                end = j

        drop_length = i - start
        if min_drop_samples <= drop_length <= max_drop_samples:
            drop = zenith - spo2_values[i].item()
            duration_seconds = drop_length / sampling_rate
            drop_rate = drop / duration_seconds

            if min_drop <= drop <= max_drop_threshold and drop_rate <= max_fall_rate:
                desaturation_events += 1
                dst_lbls[start:end] = 1
                i = end
    return desaturation_events, dst_lbls


def train_loop(train_dataloader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
               device="cpu", first_batch: int = 0, init_running_losses: list[float] = None,
               print_batch_interval: int = None, checkpoint_batch_interval: int = None,
               save_checkpoint_kwargs: dict = None, running_acc=False, progress_bar=True) -> \
        tuple[float | Any, float | None | Any]:
    # Resumes training from correct batch
    train_dataloader.sampler.first_batch_index = first_batch

    if checkpoint_batch_interval is not None and checkpoint_batch_interval > 0:
        assert save_checkpoint_kwargs is not None

    # Ensure train mode:
    model.train()
    model = model.to(device)

    unix_time_start = time.time()

    if init_running_losses is not None:
        period_losses = init_running_losses
    else:
        period_losses = []

    period_accs = []

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
              unit="batch",
              disable=not progress_bar) as tqdm_dataloader:
        for (i, data) in enumerate(tqdm_dataloader, start=first_batch):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Convert to accepted dtypes: float32, float64, int64 and maybe more but not sure
            labels = labels.type(torch.int64)

            inputs = inputs.to(device)
            labels = labels.to(device)

            if CONVERT_SPO2DESAT_TO_NORMAL:
                labels[labels == 4] = 0

            n_channels_found = inputs.shape[1]
            if n_channels_found != N_INPUT_CHANNELS:
                diff = n_channels_found - N_INPUT_CHANNELS
                assert diff > 0
                if subset_id == 0:
                    # Excess channels have been detected, exclude the first (typically SpO2 or Flow)
                    inputs = inputs[:, diff:, :]
                else:
                    # Excees channels have been detected, keep the first n_channels
                    inputs = inputs[:, :N_INPUT_CHANNELS, :]

            if CONVERT_SPO2_TO_DST_LABELS and n_channels_found > 1:
                if subset_id <= 7:
                    # SpO2 is first
                    spo2_i = 0
                else:
                    # SpO2 is last
                    spo2_i = N_INPUT_CHANNELS - 1

                if subset_id == 6:
                    rate = 64.0
                else:
                    rate = 32.0

                for b in range(inputs.shape[0]):
                    _, dst_lbls = detect_desaturations_profusion_torch(inputs[b, spo2_i, :], sampling_rate=rate)
                    inputs[b, spo2_i, :] = dst_lbls

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            if outputs.shape.numel() * N_INPUT_CHANNELS != inputs.shape.numel() * N_CLASSES \
                    and labels.shape[0] != labels.shape.numel():
                # Per window classification nut labels per sample:
                labels_by_window = torch.zeros((labels.shape[0]), device=device, dtype=torch.int64)
                for batch_index in range(labels.shape[0]):
                    labels_by_window[batch_index] = get_window_label(labels[batch_index, :])[0].to(device)
                batch_loss = criterion(outputs, labels_by_window)

                if running_acc or (i + RUNNING_LOSS_PERIOD) >= batches:
                    batch_labels = torch.ravel(labels_by_window)
            else:
                if isinstance(criterion, (monai.losses.DiceLoss,
                                          monai.losses.GeneralizedDiceLoss,
                                          monai.losses.FocalLoss)):
                    batch_loss = criterion(outputs, labels.view(labels.shape[0], 1, labels.shape[1]))
                else:
                    batch_loss = criterion(outputs, labels)
                if running_acc or (i + RUNNING_LOSS_PERIOD) >= batches:
                    batch_labels = torch.ravel(labels)

            batch_loss.backward()
            optimizer.step()

            # Compute running loss:
            period_losses.append(batch_loss.item())

            if len(period_losses) > RUNNING_LOSS_PERIOD:
                period_losses = period_losses[-RUNNING_LOSS_PERIOD:]

            running_loss = sum(period_losses) / len(period_losses)

            if running_acc or (i + RUNNING_LOSS_PERIOD) >= batches:
                _, batch_predictions = torch.max(outputs, dim=1, keepdim=False)
                batch_predictions = torch.ravel(batch_predictions)
                batch_train_acc = accuracy(batch_predictions, batch_labels, task="multiclass",
                                           num_classes=N_CLASSES).item()
                period_accs.append(batch_train_acc)
                if len(period_accs) > RUNNING_LOSS_PERIOD:
                    period_accs = period_accs[-RUNNING_LOSS_PERIOD:]
                running_acc = sum(period_accs) / len(period_accs)
            else:
                running_acc = None

            if epch is not None:
                if running_acc is not None:
                    tqdm_dataloader.set_postfix(running_loss=f"{running_loss:.5f}",
                                                running_acc=f"{running_acc:.2f}",
                                                epoch=epch)
                else:
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
                save_checkpoint(batch=i, running_losses=period_losses, running_accuracy=running_acc,
                                **save_checkpoint_kwargs)
    return running_loss, running_acc


if __name__ == "__main__":
    if COMPUTE_PLATFORM == "opencl" or COMPUTE_PLATFORM == "ocl":
        # This was the method before version 0.1.0 of dlprimitives pytorch backend: http://blog.dlprimitives.org/
        # PATH_TO_PT_OCL_DLL = Path(config["paths"]["local"]["pt_ocl_dll"])
        # PATH_TO_DEPENDENCY_DLLS = Path(config["paths"]["local"]["dependency_dlls"])
        # os.add_dll_directory(str(PATH_TO_DEPENDENCY_DLLS))
        # torch.ops.load_library(str(PATH_TO_PT_OCL_DLL))
        # device = "privateuseone:0"

        # Since 0.1.0:
        # Download appropriate wheel from: https://github.com/artyom-beilis/pytorch_dlprim/releases
        # the cp number depends on Python version
        # So for Windows, torch==2.4
        # pip install pytorch_ocl-0.1.0+torch2.4-cp310-none-linux_x86_64.whl
        import pytorch_ocl

        device = "ocl:0"
    elif torch.cuda.is_available():
        if "cuda:" in COMPUTE_PLATFORM:
            device = torch.device(COMPUTE_PLATFORM)
        else:
            device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    print(f"Using train batch size: {BATCH_SIZE}")
    print(f"Using test batch size: {BATCH_SIZE_TEST}")
    # Prepare train dataloader:
    # train_loader = get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = get_pre_batched_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pre_fetch=PRE_FETCH)
    test_loader = get_pre_batched_test_loader(batch_size=BATCH_SIZE_TEST, num_workers=NUM_WORKERS_TEST,
                                              pre_fetch=PRE_FETCH_TEST, shuffle=False)

    # Find out window size:
    sample_batch_input, sample_batch_labels = next(iter(train_loader))
    window_size = sample_batch_input.shape[2]
    N_CLASSES = int(torch.max(sample_batch_labels)) + 1
    signals_found = sample_batch_input.shape[1]
    assert signals_found >= N_INPUT_CHANNELS  # N_INPUT_CHANNELS is defined in config.yml

    if CONVERT_SPO2DESAT_TO_NORMAL:
        assert N_CLASSES == 5
        N_CLASSES = 4
    print(sample_batch_input.shape)
    print(sample_batch_labels.shape)
    print(f"Window size: {window_size}. Batch size: {BATCH_SIZE}")
    print(f"# Classes: {N_CLASSES}")
    print(f"# of signals found: {signals_found}. # of signals to use: {N_INPUT_CHANNELS}")
    print(f"Class weights: {CLASS_WEIGHTS}")

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
            net = UNet(nclass=N_CLASSES, in_chans=N_INPUT_CHANNELS, max_channels=512, depth=DEPTH, layers=LAYERS,
                       kernel_size=KERNEL_SIZE,
                       sampling_method=SAMPLING_METHOD)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "UResIncNet":
            net = UResIncNet(nclass=N_CLASSES, in_chans=N_INPUT_CHANNELS, first_out_chans=FIRST_OUT_CHANS,
                             max_channels=512, depth=DEPTH, layers=LAYERS,
                             kernel_size=KERNEL_SIZE,
                             sampling_factor=2, sampling_method=SAMPLING_METHOD, dropout=DROPOUT,
                             skip_connection=True, extra_final_conv=False, neg_slope=NEG_SLOPE,
                             custom_weight_init=CUSTOM_WEIGHT_INIT)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "AttUResIncNet":
            net = UResIncNet(nclass=N_CLASSES, in_chans=N_INPUT_CHANNELS, first_out_chans=FIRST_OUT_CHANS,
                             max_channels=512, depth=DEPTH, layers=LAYERS,
                             kernel_size=KERNEL_SIZE,
                             sampling_factor=2, sampling_method=SAMPLING_METHOD, dropout=DROPOUT,
                             skip_connection=True, extra_final_conv=False, attention=True, neg_slope=NEG_SLOPE,
                             custom_weight_init=CUSTOM_WEIGHT_INIT)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "UResIncNet-2":
            net = UResIncNet(nclass=N_CLASSES, in_chans=N_INPUT_CHANNELS, first_out_chans=FIRST_OUT_CHANS,
                             max_channels=512, depth=DEPTH, layers=LAYERS,
                             kernel_size=KERNEL_SIZE,
                             sampling_factor=2, sampling_method=SAMPLING_METHOD, dropout=DROPOUT,
                             skip_connection=True, extra_final_conv=True, neg_slope=NEG_SLOPE,
                             custom_weight_init=CUSTOM_WEIGHT_INIT)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "ConvNet":
            net = ConvNet(nclass=N_CLASSES, in_size=window_size, in_chans=N_INPUT_CHANNELS, max_channels=512,
                          depth=DEPTH,
                          layers=LAYERS,
                          kernel_size=KERNEL_SIZE, sampling_method=SAMPLING_METHOD)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "ResIncNet":
            net = ResIncNet(nclass=N_CLASSES, in_size=window_size, in_chans=N_INPUT_CHANNELS, max_channels=512,
                            depth=DEPTH,
                            layers=LAYERS,
                            kernel_size=KERNEL_SIZE, neg_slope=NEG_SLOPE,
                            sampling_factor=2, sampling_method=SAMPLING_METHOD, skip_connection=True)
            net_kwargs = net.get_kwargs()
        elif NET_TYPE == "CombinedNet":
            net = CombinedNet(nclass=N_CLASSES, in_size=window_size, in_chans=N_INPUT_CHANNELS,
                              first_out_chans=FIRST_OUT_CHANS,
                              max_channels=512,
                              depth=DEPTH,
                              kernel_size=KERNEL_SIZE, layers=LAYERS, sampling_factor=2,
                              sampling_method=SAMPLING_METHOD, dropout=DROPOUT, neg_slope=NEG_SLOPE,
                              skip_connection=True, lstm_max_features=LSTM_MAX_FEATURES, lstm_layers=LSTM_LAYERS,
                              lstm_dropout=LSTM_DROPOUT, lstm_bidirectional=True,
                              lstm_depth=1, custom_weight_init=CUSTOM_WEIGHT_INIT)

        net_kwargs = net.get_kwargs()
        initial_running_losses = None
        lr_scheduler = None
        lr_scheduler_kwargs = None
        start_from_epoch = 1
        start_from_batch = 0

        # Define loss:
        if "dl" in LOSS_FUNCTION:
            # Dice Loss family
            if weights is not None:
                loss_kwargs = {"weight": weights, "softmax": True, "reduction": "mean", "to_onehot_y": True}
            else:
                loss_kwargs = {"weight": None, "softmax": True, "reduction": "mean", "to_onehot_y": True}

            if LOSS_FUNCTION == "cel_dl":
                loss_kwargs["weight_cel"] = CEL_FL_WEIGHT
                print(f"Using combine loss with CEL weight: {CEL_FL_WEIGHT}")
                loss = CelDlLoss(**loss_kwargs)
            elif LOSS_FUNCTION == "gdl":
                loss_kwargs.pop("weight")
                loss = GeneralizedDiceLoss(**loss_kwargs)
            elif LOSS_FUNCTION == "cel_gdl":
                loss_kwargs["weight_cel"] = CEL_FL_WEIGHT
                loss_kwargs["scale_losses"] = True
                loss_kwargs["ema_scaling"] = False
                loss_kwargs["learn_weight_cel"] = LEARNABLE_CEL_WEIGHT
                print(f"Using combine loss with CEL weight: {CEL_FL_WEIGHT}")
                loss = CelGdlLoss(**loss_kwargs)
            elif LOSS_FUNCTION == "fl_dl":
                loss_kwargs["weight_fl"] = CEL_FL_WEIGHT
                loss_kwargs["gamma"] = 2.0
                print(f"Using combine loss with CEL weight: {CEL_FL_WEIGHT}")
                loss = FlDlLoss(**loss_kwargs)
            elif LOSS_FUNCTION == "fl_gdl":
                loss_kwargs["weight_fl"] = CEL_FL_WEIGHT
                loss_kwargs["gamma"] = 2.0
                print(f"Using combine loss with CEL weight: {CEL_FL_WEIGHT}")
                loss = FlGdlLoss(**loss_kwargs)
            elif "gwdl" in LOSS_FUNCTION:
                # Generalized-Wasserstein-Dice-Loss
                if N_CLASSES == 5:
                    M = torch.tensor([[0.0, 2.0, 2.0, 1.7, 0.3],
                                      [2.0, 0.0, 0.3, 0.5, 2.0],
                                      [2.0, 0.3, 0.0, 0.5, 2.0],
                                      [1.7, 0.5, 0.5, 0.0, 1.5],
                                      [0.3, 2.0, 2.0, 1.5, 0.0]], dtype=torch.float32).to(device)
                else:
                    assert N_CLASSES == 4
                    M = torch.tensor([[0.0, 2.0, 2.0, 1.7],
                                      [2.0, 0.0, 0.3, 0.5],
                                      [2.0, 0.3, 0.0, 0.5],
                                      [1.7, 0.5, 0.5, 0.0]], dtype=torch.float32).to(device)
                if use_weighted_loss:
                    loss_kwargs = {"dist_matrix": M, "weighting_mode": "GDL", "reduction": "mean"}
                else:
                    loss_kwargs = {"dist_matrix": M, "weighting_mode": "default", "reduction": "mean"}
                if LOSS_FUNCTION == "cel_gwdl":
                    loss_kwargs["weight_cel"] = CEL_FL_WEIGHT
                    loss_kwargs["learn_weight_cel"] = LEARNABLE_CEL_WEIGHT
                    loss_kwargs["weighting_mode"] = "default"
                    loss_kwargs["scale_losses"] = True
                    print(f"Using combine loss with CEL weight: {CEL_FL_WEIGHT}")
                    loss = CelGwdlLoss(**loss_kwargs)
                else:
                    loss = GeneralizedWassersteinDiceLoss(**loss_kwargs)
            else:
                loss = DiceLoss(**loss_kwargs)

        elif "fl" in LOSS_FUNCTION:
            if weights is not None:
                loss_kwargs = {"gamma": 2.0, "weight": weights, "reduction": "mean",
                               "to_onehot_y": True,
                               "use_softmax": True}
            else:
                loss_kwargs = {"gamma": 2.0, "reduction": "mean", "to_onehot_y": True,
                               "use_softmax": True}
            if LOSS_FUNCTION == "fl_gdl":
                loss_kwargs["weight_fl"] = CEL_FL_WEIGHT
                loss_kwargs["softmax"] = True
                loss_kwargs.pop("use_softmax")
                loss = FlGdlLoss(**loss_kwargs)
            else:
                loss = FocalLoss(**loss_kwargs)
        else:
            # cel
            if weights is not None:
                loss_kwargs = {"weight": weights}
                loss = nn.CrossEntropyLoss(**loss_kwargs)
            else:
                loss_kwargs = None
                loss = nn.CrossEntropyLoss()

        print(f"Using loss function: {type(loss).__name__}")

        # Set LR:
        lr = LR

        # Model should go to device first before initializing optimizer:
        net = net.to(device)

        # Define optimizer:
        if len(list(loss.parameters())) > 0:
            optim_learnable_params = list(net.parameters()) + list(loss.parameters())
        else:
            optim_learnable_params = net.parameters()
        if OPTIMIZER == "adam":
            optim_kwargs = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-08}
            optimizer = optim.Adam(optim_learnable_params, **optim_kwargs)
        else:  # sgd
            optim_kwargs = {"lr": lr, "momentum": 0.7}
            optimizer = optim.SGD(optim_learnable_params, **optim_kwargs)

    # torchinfo summary
    print(f"Subset: {subset_id}")
    print(NET_TYPE)
    print(IDENTIFIER)

    summary(net, input_size=(BATCH_SIZE, N_INPUT_CHANNELS, window_size),
            col_names=('input_size', "output_size", "kernel_size", "num_params"), device=device)

    print(optim_kwargs)

    if lr_scheduler is None:
        last_completed_epoch = start_from_epoch - 1
        if LR_WARMUP and last_completed_epoch < LR_WARMUP_DURATION:
            warmup_iters = LR_WARMUP_DURATION - last_completed_epoch
            if LR_WARMUP_ASCENDING:
                starting_factor = 0.3 + last_completed_epoch * (1 - 0.3) / LR_WARMUP_DURATION
                lr_scheduler_kwargs = {"start_factor": starting_factor, "end_factor": 1, "total_iters": warmup_iters}
                lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, **lr_scheduler_kwargs)
            else:
                starting_factor = 1 - last_completed_epoch * (1 - 0.3) / LR_WARMUP_DURATION
                lr_scheduler_kwargs = {"start_factor": starting_factor, "end_factor": 0.3, "total_iters": warmup_iters}
                lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, **lr_scheduler_kwargs)

    if EARLY_STOPPING:
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA, verbose=True)
    else:
        early_stopping = None

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
                         "batch_size": BATCH_SIZE,
                         "device": "cpu" if COMPUTE_PLATFORM == "opencl" else "infer"
                         # opencl port doc states that model should be saved on cpu
                         }
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
            last_running_loss, last_running_acc = train_loop(train_dataloader=train_loader,
                                                             model=net,
                                                             optimizer=optimizer,
                                                             criterion=loss,
                                                             device=device,
                                                             first_batch=start_from_batch,
                                                             init_running_losses=initial_running_losses,
                                                             print_batch_interval=None,
                                                             checkpoint_batch_interval=SAVE_MODEL_BATCH_INTERVAL,
                                                             save_checkpoint_kwargs=checkpoint_kwargs,
                                                             running_acc=False)

            time_elapsed = time.time() - unix_time_start
            # print(f"Epoch: {epoch} finished. Hours/Epoch: {time_elapsed / epoch / 3600}")

            if lr_scheduler is not None:
                if epoch % LR_WARMUP_STEP_EPOCH_INTERVAL == 0:
                    lr_scheduler.step()
                tqdm_epochs.set_postfix(current_base_lr=f"{lr_scheduler.get_last_lr()[0]:.5f}")

            if TESTING_EPOCH_INTERVAL is not None and epoch % TESTING_EPOCH_INTERVAL == 0:
                # print(f"Testing epoch: {epoch}")
                metrics, cm, roc_info, pr_info = test_loop(model=net, test_dataloader=test_loader, n_class=N_CLASSES,
                                                           device=device, verbose=False,
                                                           progress_bar=True)
                val_acc = metrics['aggregate_accuracy']
                val_mcc = metrics["aggregate_mcc"]
                tqdm_epochs.set_postfix(epoch_val_acc=f"{val_acc:.2f}")
                tqdm_epochs.set_postfix(epoch_val_mcc=f"{float(val_mcc):.2f}")

                # Save model:
                save_checkpoint(batch=batches_in_epoch - 1, test_metrics=metrics, test_cm=cm, roc_info=roc_info,
                                pr_info=pr_info,
                                running_loss=last_running_loss, running_accuracy=last_running_acc, **checkpoint_kwargs)

                if EARLY_STOPPING:
                    # Check early stopping condition
                    es_metric = metrics[EARLY_STOPPING_METRIC]
                    es_metric = 0 if es_metric == "nan" else es_metric
                    early_stopping.check_early_stop(es_metric)

                    if early_stopping.stop_training:
                        print(f"Early stopping at epoch {epoch}")
                        break

            elif SAVE_MODEL_EVERY_EPOCH:
                # Save model:
                save_checkpoint(batch=batches_in_epoch - 1,
                                running_loss=last_running_loss, running_accuracy=last_running_acc, **checkpoint_kwargs)

            start_from_batch = 0
            initial_running_losses = None
    print('Finished Training')
    print(datetime.datetime.now())

    checkpoint_kwargs["epoch"] = epoch
    if isinstance(train_loader.sampler.rng, torch.Generator):
        checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.get_state()
    else:
        checkpoint_kwargs["dataloader_rng_state"] = train_loader.sampler.rng.getstate()

    # Save model:
    save_checkpoint(batch=batches_in_epoch - 1, **checkpoint_kwargs)
