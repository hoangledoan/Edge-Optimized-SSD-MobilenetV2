from torchprofile import profile_macs
import torch
from torch import nn
import matplotlib.pyplot as plt

def get_model_macs(model, inputs) -> int:
    """Get number of Flops."""
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    # Filter parameters to include only those with dim > 1 (e.g., weights, not biases)
    weight_params = [(name, param) for name, param in model.named_parameters() if param.dim() > 1]
    num_plots = len(weight_params)

    # Determine grid size for subplots
    rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
    fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 2))
    axes = axes.ravel()

    # Plot each weight distribution
    for plot_index, (name, param) in enumerate(weight_params):
        ax = axes[plot_index]
        param_data = param.detach().view(-1).cpu()
        if count_nonzero_only:
            param_data = param_data[param_data != 0]
        ax.hist(param_data, bins=bins, density=True, color='blue', alpha=0.5)
        ax.set_xlabel(name)
        ax.set_ylabel('Density')

    # Hide any unused axes
    for ax in axes[num_plots:]:
        ax.axis('off')

    # Adjust layout and show plot
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig("weight_distribution.jpg")

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB