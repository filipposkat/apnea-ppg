import yaml
from pathlib import Path
import datetime, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Local imports:
from data_loaders_iterable import get_saved_train_loader, IterDataset
from UNet import UNet

# --- START OF CONSTANTS --- #
EPOCHS = 10
BATCH_SIZE = 256

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    PATH_TO_SUBSET1 = Path(config["paths"]["local"]["subset_1_directory"])
else:
    PATH_TO_OBJECTS = Path(__file__).parent.joinpath("data", "serialized-objects")
    PATH_TO_SUBSET1 = Path(__file__).parent.joinpath("data", "subset-1")

# --- END OF CONSTANTS --- #

models_path = PATH_TO_SUBSET1.joinpath("saved-models")
models_path.mkdir(parents=True, exist_ok=True)


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
    torch.save(state, model_path.joinpath(f"{identifier}.pt"))


def load_model(net_type: str, identifier: int):
    model_path = models_path.joinpath(f"{net_type}", f"{identifier}.pt")
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


# Prepare train dataloader:
train_loader = get_saved_train_loader(batch_size=BATCH_SIZE)

# Create Network:
unet = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=3, sampling_method="pooling")

# Define loss and optimizer:
ce_loss = nn.CrossEntropyLoss()
optim_kwargs = {"lr": 0.01, "momentum": 0.7}
sgd = optim.SGD(unet.parameters(), **optim_kwargs)

# Check for device:
if torch.cuda.is_available():
    device = torch.device("cuda")
    unet = unet.to(device)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    unet = unet.to(device)
else:
    device = torch.device("cpu")

print(f"Device: {device}")


def train_loop(net, optimizer, criterion, epochs=EPOCHS, save_model_every_epoch: bool = False, identifier: int = None):
    print(datetime.datetime.now())
    for epoch in range(epochs):  # loop over the dataset multiple times
        loader = train_loader
        batches = len(loader)
        # print(f"Batches in epoch: {batches}")

        unix_time_start = time.time()

        running_loss = 0.0
        for (i, data) in tqdm(enumerate(iter(loader)), total=batches):
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

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 9999:  # print every 1000 mini-batches
                time_elapsed = time.time() - unix_time_start

                print(f'[Epoch:{epoch + 1:3d}/{epochs:3d}, Batch{i + 1:6d}/{batches:6d}]'
                      f' Running Avg loss: {running_loss / 2000:.3f}'
                      f' Minutes/Batch: {time_elapsed / (i + 1) / 60}')

                running_loss = 0.0
                if save_model_every_epoch:
                    save_model_state(unet, optimizer=optimizer, optimizer_kwargs=optim_kwargs,
                                     criterion=criterion, net_type="UNET", identifier=identifier,
                                     batch_size=BATCH_SIZE, epoch=epoch, other_details=f"Batch: {i}")
        time_elapsed = time.time() - unix_time_start
        print(f"Epoch: {epoch} finished. Hours/Epoch: {time_elapsed / (epochs + 1) / 3600}")

    print('Finished Training')
    print(datetime.datetime.now())


# Train:
train_loop(unet, sgd, ce_loss, epochs=EPOCHS, save_model_every_epoch=True, identifier=1)

# Save model:
save_model_state(unet, optimizer=sgd, optimizer_kwargs=optim_kwargs,
                 criterion=ce_loss, net_type="UNET", identifier=1,
                 batch_size=BATCH_SIZE, epoch=EPOCHS)
