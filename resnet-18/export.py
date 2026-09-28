# System functionality like creating directories and reading env-vars
import os

# Configuration parsing
import configparser

# Save verification input-output pair as numpy array
import numpy as np

# PyTorch base package: Math and Tensor Stuff
import torch

# Loads shuffled batches from datasets
from torch.utils.data import DataLoader

# PyTorch vision datasets and transformations
from torchvision import datasets, transforms

# Export brevitas quantized models to QONNX dialect
from brevitas.export import export_qonnx, export_onnx_qcdq

# The Vision classification model
from brevitas_examples.bnn_pynq.models.resnet import quant_resnet18
import brevitas.nn as qnn

# Seeding RNGs for reproducibility
from utils import seed

# Path to the CIFAR-100 dataset
CIFAR100_ROOT = os.environ.setdefault(
    "CIFAR100_ROOT", "/scratch/hpc-prf-ekiapp/datasets/cifar100/data"
)

# Export function mapping
EXPORTERS = {"qonnx": export_qonnx, "qcdq": export_onnx_qcdq}

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.inpQuantizer = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.resnet = quant_resnet18(config)

    def forward(self, x):
        x = self.inpQuantizer(x)
        return self.resnet(x)


# Exports the model to ONNX in conjunction with an input-output pair for
# verification
def export(model, batch_size, format="qonnx", **kwargs):
    # Do the forward pass for generating verification data and tracing the model
    # for export on CPU only
    device = "cpu"
    # Move the model to the training device
    model = model.to(device)  # noqa: Shadows model...
    # Set model to evaluation mode
    model = model.eval()  # noqa: Shadows model...

    # Transformation to be applied to the input images: Basic preprocessing
    # turning images into tensors and normalizing (no augmentation for export)
    tf = transforms.Compose(
        [
            # Convert from PIL image to PyTorch tensors
            transforms.ToTensor(),
            # CIFAR-100 statistics on the whole training set
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

    # Load the CIFAR-100 test split (should already be in CIFAR100_ROOT,
    # otherwise download)
    dataset = datasets.CIFAR100(CIFAR100_ROOT, False, download=True, transform=tf)
    # Create a batched data loader for the test split
    export_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Sample the first batch from the export dataset
    inp, cls = next(iter(export_data))

    # Also save the model output predictions (probabilities)
    with torch.no_grad():
        out = model(inp)

    # Export the model to ONNX using the input example
    EXPORTERS[format](model, (inp,), "outputs/resnet/model.onnx", **kwargs)

    # Save the input and output data for verification purposes later
    np.save("outputs/resnet/inp.npy", inp.numpy())
    np.save("outputs/resnet/out.npy", out.numpy())
    np.save("outputs/resnet/cls.npy", cls.numpy())


# Export configuration parameters
params = {
    "export": {
        # Batch size for export data
        "batch_size": 1,
        # Export format: qonnx or qcdq
        "format": "qonnx",
    }
}

# Script entrypoint
if __name__ == "__main__":
    # Read the model configuration
    config = configparser.ConfigParser()
    config.read("config.ini")
    # Seed all RNGs
    seed(int(config["MODEL"]["SEED"]))
    # Create a new model instance according to the configuration
    model = Model()
    # Load the trained model parameters
    model.load_state_dict(torch.load("outputs/resnet/model.pt"))
    # Create the output directory if it does not already exist
    os.makedirs("outputs/resnet", exist_ok=True)
    # Pass the model and the export configuration to the export function
    export(model, **params["export"])
