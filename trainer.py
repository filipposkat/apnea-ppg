import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

# Local imports:
from data_loaders_iterable import get_saved_train_loader
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
train_loader = get_saved_train_loader()

# Create Network:
unet = UNet(nclass=5, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=3, sampling_method="pooling")
ce_loss = nn.CrossEntropyLoss()
sgd = optim.SGD(unet.parameters(), lr=0.01, momentum=0.7)


def train_loop(net, criterion, optimizer, epochs=EPOCHS):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

# Train:
train_loop(unet, ce_loss, sgd, epochs=10)

# Save model:
crit = "ce"
opt = "sgd"
model_path = models_path.joinpath(f"UNET {unet.get_parameter_summary()}  -  Criterion {crit} - Optimizer {opt}")
torch.save(unet.state_dict(), model_path)
