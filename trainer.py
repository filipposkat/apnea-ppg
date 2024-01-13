import yaml
from pathlib import Path
import datetime, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchinfo import summary

from tqdm import tqdm

# Local imports:
from data_loaders_iterable import IterDataset, worker_init_fn, get_saved_train_loader
from pre_batched_dataloader import get_pre_batched_train_loader

from UNet import UNet
from tester import test_loop

# --- START OF CONSTANTS --- #
EPOCHS = 100
BATCH_SIZE = 256
NUM_WORKERS = 4
LR_TO_BATCH_RATIO = 1 / 25600
LR_WARMUP = True
SAVE_MODEL_EVERY_EPOCH = True
TESTING_EPOCH_INTERVAL = 1

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
    PATH_TO_SUBSET1_TRAINING = Path(config["paths"]["local"]["subset_1_training_directory"])
else:
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")
    PATH_TO_SUBSET1_TRAINING = PATH_TO_SUBSET1

# --- END OF CONSTANTS --- #

models_path = PATH_TO_SUBSET1_TRAINING.joinpath("saved-models")
models_path.mkdir(parents=True, exist_ok=True)

# Check for device:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


def save_model_state(net, optimizer, optimizer_kwargs, criterion,
                     net_type: str, identifier: int,
                     batch_size: int, epoch: int, other_details: str = ""):
    model_path = models_path.joinpath(f"{net_type}")
    model_path.mkdir(parents=True, exist_ok=True)
    # identifier = 1
    # while net_path.joinpath(f"{identifier}").is_dir():
    #     identifier += 1

    txt_path = model_path.joinpath(f"{identifier}_details.txt")
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
        'net_class': net.__class__,
        'net_state_dict': net.state_dict(),
        'net_kwargs': net.get_kwargs(),
        'optimizer_class': optimizer.__class__,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_kwargs': optimizer_kwargs,
        'criterion': criterion
    }
    torch.save(state, model_path.joinpath(f"id:{identifier}-epoch:{epoch}.pt"))


def load_model(net_type: str, identifier: int, epoch: int):
    model_path = models_path.joinpath(f"{net_type}", f"id:{identifier}-epoch:{epoch}.pt")
    state = torch.load(model_path)
    net_class = state["net_class"]
    net_state = state["net_state_dict"]
    net_kwargs = state["net_kwargs"]
    optimizer_class = state["optimizer_class"]
    optimizer_state_dict = state["optimizer_state_dict"]
    optimizer_kwargs = state["optimizer_kwargs"]
    criterion = state["criterion"]

    net = net_class(**net_kwargs)
    net.load_state_dict(net_state)

    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(optimizer_state_dict)

    return net, optimizer, criterion


def train_loop(train_loader, net, optimizer, criterion, lr_scheduler=None, lr_step_batch_interval: int = 10000,
               device=device):

    # Ensure train mode:
    net.train()

    unix_time_start = time.time()

    running_loss = 0.0
    batches = len(train_loader)
    for (i, data) in tqdm(enumerate(train_loader), total=batches):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        if lr_scheduler and i % lr_step_batch_interval == i - 1:
            lr_scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 10000 == 9999:  # print every 10000 mini-batches
            time_elapsed = time.time() - unix_time_start

            print(f'[Batch{i + 1:7d}/{batches:7d}]'
                  f' Running Avg loss: {running_loss / 2000:.3f}'
                  f' Minutes/Batch: {time_elapsed / (i + 1) / 60}')

            running_loss = 0.0


if __name__ == "__main__":
    # Prepare train dataloader:
    # train_loader = get_saved_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = get_pre_batched_train_loader(batch_size=BATCH_SIZE, n_workers=NUM_WORKERS)

    # Create Network:
    unet = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=3, sampling_method="pooling")
    unet = unet.to(device)

    summary(unet, input_size=(BATCH_SIZE, 1, 512),
            col_names=('input_size', "output_size", "kernel_size", "num_params"), device=device)

    # Define loss and optimizer:
    ce_loss = nn.CrossEntropyLoss()
    lr = LR_TO_BATCH_RATIO * BATCH_SIZE
    optim_kwargs = {"lr": 0.01, "momentum": 0.7}
    print(optim_kwargs)
    sgd = optim.SGD(unet.parameters(), **optim_kwargs)
    if LR_WARMUP:
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=sgd, start_factor=0.3, end_factor=1, total_iters=3)
    else:
        lr_scheduler = None

    # Train:
    print(datetime.datetime.now())
    unix_time_start = time.time()
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        # print(f"Batches in epoch: {batches}")
        train_loop(train_loader=train_loader, net=unet, optimizer=sgd, criterion=ce_loss,
                   device=device)

        time_elapsed = time.time() - unix_time_start
        print(f"Epoch: {epoch} finished. Hours/Epoch: {time_elapsed / (EPOCHS + 1) / 3600}")

        if lr_scheduler:
            lr_scheduler.step()

        if SAVE_MODEL_EVERY_EPOCH:
            # Save model:
            save_model_state(unet, optimizer=sgd, optimizer_kwargs=optim_kwargs,
                             criterion=ce_loss, net_type="UNET", identifier=1,
                             batch_size=BATCH_SIZE, epoch=epoch)

        if epoch % TESTING_EPOCH_INTERVAL == TESTING_EPOCH_INTERVAL - 1:
            test_loop(net=unet)

    print('Finished Training')
    print(datetime.datetime.now())

    # Save model:
    save_model_state(unet, optimizer=sgd, optimizer_kwargs=optim_kwargs,
                     criterion=ce_loss, net_type="UNET", identifier=1,
                     batch_size=BATCH_SIZE, epoch=EPOCHS)
