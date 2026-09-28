# System functionality like creating directories and reading env-vars
import os

# YAML for saving experiment metrics
import yaml

# Progressbar in for loops
import tqdm

# PyTorch base package: Math and Tensor Stuff
import torch

# Loads shuffled batches from datasets
from torch.utils.data import DataLoader, random_split

# PyTorch vision datasets and transformations
from torchvision import datasets, transforms

import configparser

# The Vision classification model
from brevitas_examples.bnn_pynq.models.resnet import quant_resnet18
import brevitas.nn as qnn

# Seeding RNGs for reproducibility, configuration of optimizer and loss
from utils import seed, get_optimizer, get_criterion

# Path to the CIFAR-100 dataset
CIFAR100_ROOT = os.environ.setdefault(
    "CIFAR100_ROOT", "/scratch/hpc-prf-ekiapp/datasets/cifar100/data"
)


# Main training loop: Takes a model, loads the dataset and sets up the
# optimizer. Runs the configured number of training epochs
def train(
    model,
    batch_size,
    epochs,
    criterion,
    optimizer,
    loader,  # noqa
    scheduler,
):
    # Check whether GPU training is available and select the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to training mode
    model = model.train()  # noqa: Shadows model...

    # Get the optimizer and register the model parameters
    optimizer = get_optimizer(  # noqa: Shadows optimizer...
        **optimizer, parameters=model.parameters()
    )
    # Get the optimization criterion instance
    criterion = get_criterion(criterion)

    # Learning rate scheduler reducing the learning rate if the validation loss
    # stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler)

    # Transformation to be applied to the input images: Rather basic
    # preprocessing turning images into tensors and normalizing with only
    # minimal data augmentation
    tf = transforms.Compose(
        [
            # Convert from PIL image to PyTorch tensors
            transforms.ToTensor(),
            # Random horizontal flip in 50% of the cases
            transforms.RandomHorizontalFlip(),
            # Radom crop of the image resized back to 32x32 pixels
            transforms.RandomResizedCrop((32, 32)),
            # CIFAR-10 statistics on the whole training set
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

    # Load the Vision training split (should already be in CIFAR100_ROOT,
    # otherwise download)
    dataset = datasets.CIFAR100(CIFAR100_ROOT, True, download=True, transform=tf)
    # Split into 90% training and 10% validation dataset
    train_data, valid_data = random_split(dataset, [0.9, 0.1])

    # Create a batched and shuffled data loader for each of the dataset splits
    train_data = DataLoader(train_data, batch_size=batch_size, **loader)
    valid_data = DataLoader(valid_data, batch_size=batch_size, **loader)

    # Collect training and validation loss and learning rate per epoch
    _loss, _lr = [], []
    # Collect training and validation accuracy per epoch
    _acc = []

    # Run the configured number of training epochs
    for _ in tqdm.trange(epochs, desc="epoch"):
        # Collect training and validation loss per epoch
        train_loss, valid_loss = (0, 0)
        # Collect training and validation accuracy per epoch
        train_correct, valid_correct = (0, 0)
        train_total, valid_total = (0, 0)
        # Set model to training mode
        model = model.train()  # noqa: Shadows model...
        # Iterate the batches of (input, target labels) pairs
        for x, y in tqdm.tqdm(train_data, desc="train-batch", leave=False):
            # Clear gradients of last iteration
            optimizer.zero_grad(set_to_none=True)
            # Feed input data to model to get predictions
            p = model(x.to(device))  # noqa: Duplicate, see below
            # Loss between class probabilities and true class labels
            loss = criterion(p, y.to(device))  # noqa: Shadows outer scope
            # Backpropagation of the error to compute gradients
            loss.backward()
            # Parameter update step
            optimizer.step()
            # Accumulate the loss over the whole validation dataset
            train_loss += loss.item()
            # Calculate accuracy: compare predicted class with true labels
            pred = p.argmax(dim=1)
            train_correct += (pred == y.to(device)).sum().item()
            train_total += y.size(0)
        # Clear gradients of last iteration
        optimizer.zero_grad(set_to_none=True)
        # Switch the model to evaluation mode, disabling dropouts and scale
        # calibration
        model = model.eval()  # noqa: Shadows model...
        # Validation requires no gradients
        with torch.no_grad():
            # Iterate the batches of (input, target labels) pairs
            for x, y in tqdm.tqdm(valid_data, "valid-batch", leave=False):
                # Feed input data to model to get predictions
                p = model(x.to(device))  # noqa: Duplicate, see above
                # Loss between class probabilities and true class labels
                loss = criterion(p, y.to(device))  # noqa: Shadows outer scope
                # Accumulate the loss over the whole validation dataset
                valid_loss += loss.item()
                # Calculate accuracy: compare predicted class with true labels
                pred = p.argmax(dim=1)
                valid_correct += (pred == y.to(device)).sum().item()
                valid_total += y.size(0)

        # Calculate accuracy percentages
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        valid_acc = 100.0 * valid_correct / valid_total if valid_total > 0 else 0.0

        # Adjust the learning rate if necessary
        scheduler.step(valid_loss)
        # Append loss information to the log
        _loss.append({"train": train_loss, "valid": valid_loss})
        # Append accuracy information to the log
        _acc.append({"train": train_acc, "valid": valid_acc})
        # keep track of the learning rate
        _lr.append({"last": scheduler.get_last_lr()})
    # Clear the gradients of last iteration
    optimizer.zero_grad(set_to_none=True)
    # Return the model, the optimizer state and the log after training
    return model.cpu(), optimizer, {"loss": _loss, "accuracy": _acc}, {"lr": _lr}


# Training configuration parameters
params = {
    "train": {
        # Number of training epochs to run
        "epochs": 500,
        # Batch size for training
        "batch_size": 1024,
        # Optimizer configuration
        "optimizer": {
            # Name of the optimization algorithm to use: adam or sgd
            "algorithm": "adam",
            # (Initial) Learning rate
            "lr": 0.001,
            # L2 regularization
            "weight_decay": 1.0e-3,
            # Coefficients used for computing running averages of gradient and its square
            "betas": [0.9, 0.999],
            # Term added to the denominator to improve numerical stability
            "eps": 1.0e-8,
        },
        # Loss function to use: cross-entropy or nll
        "criterion": "cross-entropy",
        # Learning rate scheduler (ReduceLROnPlateau) configuration
        "scheduler": {
            # Reduce learning rate by this factor if the loss stops improving
            "factor": 0.5,
            # Number of epochs without improvement to wait before reducing the learning rate
            "patience": 10,
        },
        # DataLoader keyword arguments
        "loader": {
            # Reshuffle data every epoch
            "shuffle": True,
            # Number of workers to use for loading the data in parallel
            "num_workers": 16,
            # Number of batches loaded in advance by each worker
            "prefetch_factor": 4,
            # Keep worker processes alive after consuming the dataset once
            "persistent_workers": False,
            # Load tensors into device pinned memory
            "pin_memory": True,
            # Drop the last batch if it is incomplete, i.e., smaller than the batch size
            "drop_last": True,
        },
    }
}

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.inpQuantizer = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.resnet = quant_resnet18(config)

    def forward(self, x):
        x = self.inpQuantizer(x)
        return self.resnet(x)

# Script entrypoint
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    # Seed all RNGs
    seed(int(config["MODEL"]["SEED"]))
    # Create a new model instance according to the configuration
    model = Model()
    # Pass the model and the training configuration to the training loop
    model, optimizer, metrics, lr = train(model, **params["train"])
    # Create the output directory if it does not already exist
    os.makedirs("outputs/resnet", exist_ok=True)
    # Save the model in PyTorch format
    torch.save(model.state_dict(), "outputs/resnet/model.pt")
    # Save the optimizer state in PyTorch format
    torch.save(optimizer.state_dict(), "outputs/resnet/optimizer.pt")
    # Save the training loss log as YAML
    with open("outputs/resnet/loss.yaml", "w") as file:
        # Dump the training loss log dictionary as YAML into the file
        yaml.safe_dump({"loss": metrics["loss"]}, file)
    # Save the training accuracy log as YAML
    with open("outputs/resnet/accuracy.yaml", "w") as file:
        # Dump the training accuracy log dictionary as YAML into the file
        yaml.safe_dump({"accuracy": metrics["accuracy"]}, file)
    # Save the training learning rate log as YAML
    with open("outputs/resnet/lr.yaml", "w") as file:
        # Dump the training log dictionary as YAML into the file
        yaml.safe_dump(lr, file)
