"""Data utilities for machine learning projects."""

import numpy as np
import torch


def create_random_data(n_samples=100, n_features=10, random_seed=42):
    """Create random data for demonstration purposes.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # Generate labels (a simple rule: the class is determined by the sum of features)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if np.sum(X[i]) > 0.5:
            y[i] = 1
        else:
            y[i] = 0

    return X, y


def numpy_to_torch(numpy_array, dtype=torch.float32, requires_grad=False):
    """Convert a numpy array to a PyTorch tensor.

    Args:
        numpy_array: Numpy array to convert
        dtype: PyTorch data type
        requires_grad: Whether the tensor requires gradients

    Returns:
        PyTorch tensor
    """
    return torch.tensor(numpy_array, dtype=dtype, requires_grad=requires_grad)


def get_device():
    """Get the device to use for PyTorch computations.

    Returns:
        PyTorch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
