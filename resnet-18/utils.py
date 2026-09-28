# Python's builtin random number generators
import random
# Seed the numpy random number generator as well
import numpy as np

# PyTorch base package: Math and Tensor Stuff, and an RNG as well
import torch


# Seeds all relevant random number generators to the same seed for
# reproducibility
def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# Gets an optimizer instance according to configuration and register the model
# parameters
def get_optimizer(algorithm, parameters, **kwargs):
    # Supported optimizer algorithms
    algorithms = {
        "adam": torch.optim.Adam, "sgd": torch.optim.SGD
    }
    # The configured algorithm must be among the supported ones
    assert algorithm in algorithms, f"Optimizer {algorithm} is not supported"
    # Create the optimizer instance forwarding additional arguments and
    # registering the model parameters
    return algorithms[algorithm](parameters, **kwargs)


# Gets the loss functions instance according to configuration
def get_criterion(criterion, **kwargs):
    # Supported optimization criteria
    criteria = {
        "cross-entropy": torch.nn.CrossEntropyLoss, "nll": torch.nn.NLLLoss
    }
    # The configured criterion must be among the supported ones
    assert criterion in criteria, f"Criterion {criterion} is not supported"
    # Create the loss function instance forwarding additional arguments
    return criteria[criterion](**kwargs)


# Check whether a layer is a normalization layer of some supported type
def is_norm_layer(module: torch.nn.Module) -> bool:
    # Set of normalization layer (bases) which maybe need to be patched
    norm_layers = {
        # All BatchNorm and InstanceNorm variants derive from this baseclass
        torch.nn.modules.batchnorm._NormBase,  # noqa: Access to _NormBase
        # LayerNorm has a unique implementation
        torch.nn.LayerNorm,
    }
    # Check the module against all supported norm layer types
    return any(isinstance(module, norm) for norm in norm_layers)


# Fixes export issues of normalization layers with disabled affine parameters.
def patch_missing_affine_norms(model: torch.nn.Module) -> torch.nn.Module:
    # Iterate all modules in the model container
    for name, module in model.named_modules():
        # If the module is a normalization layer it might require patching the
        # affine parameters
        if is_norm_layer(module):
            # Check whether affine scale parameters are missing
            if hasattr(module, "weight") and module.weight is None:
                # Access to running statistics can be used as reference shape to
                # patch the scales
                if hasattr(module, "running_var"):
                    module.weight = torch.nn.Parameter(
                        torch.ones_like(module.running_var)
                    )
                # If the module explicitly knows its normalized shape, this can
                # be used as a reference to patch the scales
                elif hasattr(module, "normalized_shape"):
                    module.weight = torch.nn.Parameter(
                        torch.ones(module.normalized_shape)
                    )
            # Check whether affine bias parameters are missing
            if hasattr(module, "bias") and module.bias is None:
                # Access to running statistics can be used as reference shape to
                # patch the bias
                if hasattr(module, "running_mean"):
                    module.bias = torch.nn.Parameter(
                        torch.zeros_like(module.running_var)
                    )
                # If the module explicitly knows its normalized shape, this can
                # be used as a reference to patch the bias
                elif hasattr(module, "normalized_shape"):
                    module.weight = torch.nn.Parameter(
                        torch.zeros(module.normalized_shape)
                    )
    # Return the patched model container
    return model