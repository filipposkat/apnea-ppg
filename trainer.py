import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

# Local imports:
import data_loaders_iterable
from data_loaders_iterable import get_new_train_loader, get_saved_train_loader, train_array_loader, IterableDataset
from UNet import UNet

# --- START OF CONSTANTS --- #
EPOCHS = 10

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

# Prepare train dataloader:
train_loader = get_saved_train_loader(batch_size=128)

# Create Network:
unet = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=3, sampling_method="pooling")

# Define loss and optimizer:
ce_loss = nn.CrossEntropyLoss()
sgd = optim.SGD(unet.parameters(), lr=0.01, momentum=0.7)

# Check for device:
if torch.cuda.is_available():
    device = torch.device("cuda")
    unet = unet.to(device)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    unet = unet.to(device)
else:
    device = torch.device("cpu")


def train_loop(net, criterion, optimizer, epochs=EPOCHS):
    for epoch in range(epochs):  # loop over the dataset multiple times
        loader = train_loader
        batches = len(loader)
        # print(f"Batches in epoch: {batches}")

        running_loss = 0.0
        for (i, data) in enumerate(iter(loader)):
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[Epoch:{epoch + 1:3d}/{epochs:3d}, Batch{i + 1:6d}/{batches:6d}]'
                      f' Running Avg loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


# Train:
train_loop(unet, ce_loss, sgd, epochs=10)

# Save model:
crit = "ce"
opt = "sgd"
model_path = models_path.joinpath(f"UNET {unet.get_parameter_summary()}  -  Criterion {crit} - Optimizer {opt}")
torch.save(unet.state_dict(), model_path)
