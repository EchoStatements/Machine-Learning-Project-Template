"""Visualization utilities for machine learning projects."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_matrix(matrix, title="Matrix Heatmap", cmap="viridis", ax=None):
    """Plot a matrix as a heatmap.

    Args:
        matrix: 2D numpy array or PyTorch tensor
        title: Title for the plot
        cmap: Colormap to use
        ax: Matplotlib axis to plot on

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Convert PyTorch tensor to numpy if needed
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    im = ax.imshow(matrix, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

    return ax


def plot_line_chart(data, labels=None, title="Line Chart", xlabel="X", ylabel="Y", ax=None):
    """Plot a line chart.

    Args:
        data: List of data series to plot
        labels: Labels for each series
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ax: Matplotlib axis to plot on

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if not isinstance(data, list):
        data = [data]

    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(data))]

    for i, series in enumerate(data):
        # Convert PyTorch tensor to numpy if needed
        if isinstance(series, torch.Tensor):
            series = series.detach().cpu().numpy()

        ax.plot(series, label=labels[i])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return ax


def plot_bar_chart(data, labels=None, title="Bar Chart", xlabel="X", ylabel="Y", ax=None):
    """Plot a bar chart.

    Args:
        data: Data to plot
        labels: Labels for each bar
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ax: Matplotlib axis to plot on

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Convert PyTorch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    x = np.arange(len(data))
    ax.bar(x, data)

    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def create_subplots(n_rows=1, n_cols=2, figsize=(12, 8)):
    """Create a grid of subplots.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        figsize: Figure size

    Returns:
        Figure and array of axes
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.tight_layout()
    return fig, axes
