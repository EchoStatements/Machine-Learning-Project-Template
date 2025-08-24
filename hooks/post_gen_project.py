#!/usr/bin/env python
"""Post-generation script for the cookiecutter template."""

import os
import shutil
import subprocess
from pathlib import Path


def create_empty_dirs():
    """Create empty directories that might not be included in version control."""
    dirs = [
        "data",
        "models",
        "figures",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Create a .gitkeep file to ensure the directory is included in version control
        with open(os.path.join(d, ".gitkeep"), "w") as f:
            f.write("")


def create_gitignore():
    """Create a .gitignore file with common patterns for ML projects."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.tsv
*.xlsx
*.db
*.sqlite3

# Model files
*.pt
*.pth
*.h5
*.pkl
*.joblib

# Logs
logs/
*.log

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)


def create_example_notebook():
    """Create an example Jupyter notebook based on the ml_example.py file."""
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Machine Learning Example Notebook\\n",
    "\\n",
    "This notebook demonstrates basic ML operations with numpy, torch, and transformers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\\n",
    "import numpy as np\\n",
    "import torch\\n",
    "import matplotlib.pyplot as plt\\n",
    "from transformers import BertConfig, BertModel\\n",
    "\\n",
    "# Import from our project\\n",
    "from src.{{ cookiecutter.project_slug }}.data import create_random_data, numpy_to_torch, get_device\\n",
    "from src.{{ cookiecutter.project_slug }}.model import SimpleNN, create_small_transformer, apply_model\\n",
    "from src.{{ cookiecutter.project_slug }}.train import plot_matrix, plot_line_chart, plot_bar_chart, create_subplots\\n",
    "\\n",
    "# Set up matplotlib for inline display\\n",
    "%matplotlib inline\\n",
    "plt.style.use('ggplot')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Numpy Calculations\\n",
    "\\n",
    "Let's perform some basic calculations using numpy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create random data\\n",
    "X, y = create_random_data(n_samples=100, n_features=10)\\n",
    "print(f\\"Features shape: {X.shape}\\")\\n",
    "print(f\\"Labels shape: {y.shape}\\")\\n",
    "\\n",
    "# Show a sample\\n",
    "print(f\\"Sample features:\\n{X[:5, :5]}\\")\\n",
    "print(f\\"Sample labels: {y[:10]}\\")\\n",
    "\\n",
    "# Matrix operations\\n",
    "eigenvalues, eigenvectors = np.linalg.eig(X[:5, :5])\\n",
    "print(f\\"\\nEigenvalues:\\n{eigenvalues}\\")\\n",
    "\\n",
    "# Statistical operations\\n",
    "print(f\\"\\nMean of each column: {np.mean(X, axis=0)[:5]}...\\")\\n",
    "print(f\\"Standard deviation of each column: {np.std(X, axis=0)[:5]}...\\")\\n",
    "\\n",
    "# Class distribution\\n",
    "unique, counts = np.unique(y, return_counts=True)\\n",
    "print(f\\"\\nClass distribution: {dict(zip(unique, counts))}\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. PyTorch Calculations\\n",
    "\\n",
    "Now let's perform similar calculations using PyTorch."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert numpy arrays to PyTorch tensors\\n",
    "X_tensor = numpy_to_torch(X)\\n",
    "y_tensor = numpy_to_torch(y, dtype=torch.long)\\n",
    "\\n",
    "print(f\\"X tensor shape: {X_tensor.shape}\\")\\n",
    "print(f\\"y tensor shape: {y_tensor.shape}\\")\\n",
    "\\n",
    "# Move to GPU if available\\n",
    "device = get_device()\\n",
    "X_tensor = X_tensor.to(device)\\n",
    "y_tensor = y_tensor.to(device)\\n",
    "print(f\\"Using device: {device}\\")\\n",
    "\\n",
    "# Create a simple neural network\\n",
    "model = SimpleNN(input_dim=X.shape[1], hidden_dim=20, output_dim=2)\\n",
    "model = model.to(device)\\n",
    "\\n",
    "# Forward pass\\n",
    "outputs = model(X_tensor)\\n",
    "print(f\\"Model output shape: {outputs.shape}\\")\\n",
    "print(f\\"First few outputs:\\n{outputs[:5]}\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Visualization with Matplotlib\\n",
    "\\n",
    "Let's create some visualizations of our data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a figure with subplots\\n",
    "fig, axes = create_subplots(2, 2, figsize=(12, 8))\\n",
    "\\n",
    "# Plot 1: Heatmap of numpy matrix\\n",
    "plot_matrix(X[:10, :10], title='Feature Matrix Heatmap', ax=axes[0, 0])\\n",
    "\\n",
    "# Plot 2: Line plot of numpy data\\n",
    "data_series = [X[:20, i] for i in range(5)]  # First 5 features, first 20 samples\\n",
    "labels = [f'Feature {i+1}' for i in range(5)]\\n",
    "plot_line_chart(data_series, labels, title='Feature Values', \\n",
    "                xlabel='Sample Index', ylabel='Value', ax=axes[0, 1])\\n",
    "\\n",
    "# Plot 3: Heatmap of model outputs\\n",
    "plot_matrix(outputs[:10].detach().cpu().numpy(), \\n",
    "            title='Model Outputs Heatmap', cmap='plasma', ax=axes[1, 0])\\n",
    "\\n",
    "# Plot 4: Bar chart of class distribution\\n",
    "plot_bar_chart(counts, labels=unique, title='Class Distribution',\\n",
    "               xlabel='Class', ylabel='Count', ax=axes[1, 1])\\n",
    "\\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Loading a Transformers Model from Custom Config\\n",
    "\\n",
    "Now let's create a small untrained BERT model from a custom configuration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a small transformer model\\n",
    "model, config = create_small_transformer(\\n",
    "    vocab_size=1000,          # Reduced vocabulary size\\n",
    "    hidden_size=64,           # Smaller hidden size\\n",
    "    num_layers=2,             # Fewer layers\\n",
    "    num_heads=2               # Fewer attention heads\\n",
    ")\\n",
    "\\n",
    "# Print the configuration\\n",
    "print(\\"Custom BERT Configuration:\\")\\n",
    "print(config)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print model summary\\n",
    "print(f\\"Model parameters: {sum(p.numel() for p in model.parameters()):,}\\")\\n",
    "\\n",
    "# List some of the model's modules\\n",
    "for name, module in list(model.named_modules())[:10]:  # Show only first 10 modules\\n",
    "    print(f\\"{name}: {module.__class__.__name__}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a sample input\\n",
    "batch_size = 2\\n",
    "seq_length = 10\\n",
    "input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))\\n",
    "attention_mask = torch.ones(batch_size, seq_length)\\n",
    "\\n",
    "print(f\\"Input IDs shape: {input_ids.shape}\\")\\n",
    "print(f\\"Attention mask shape: {attention_mask.shape}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the model\\n",
    "model.eval()\\n",
    "with torch.no_grad():\\n",
    "    outputs = model(input_ids=input_ids, attention_mask=attention_mask)\\n",
    "\\n",
    "# Examine the outputs\\n",
    "last_hidden_state = outputs.last_hidden_state\\n",
    "pooler_output = outputs.pooler_output\\n",
    "\\n",
    "print(f\\"Last hidden state shape: {last_hidden_state.shape}\\")\\n",
    "print(f\\"Pooler output shape: {pooler_output.shape}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the embeddings\\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\\n",
    "\\n",
    "# Plot the last hidden state for the first sequence\\n",
    "plot_matrix(last_hidden_state[0], title='Last Hidden State (First Sequence)',\\n",
    "            ax=axes[0])\\n",
    "axes[0].set_xlabel('Hidden Dimension')\\n",
    "axes[0].set_ylabel('Sequence Position')\\n",
    "\\n",
    "# Plot the pooler output\\n",
    "plot_bar_chart(pooler_output[0], title='Pooler Output (First Sequence)',\\n",
    "               xlabel='Hidden Dimension', ylabel='Value', ax=axes[1])\\n",
    "\\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion\\n",
    "\\n",
    "In this notebook, we've demonstrated:\\n",
    "\\n",
    "1. Basic calculations with NumPy\\n",
    "2. Similar operations with PyTorch\\n",
    "3. Data visualization with Matplotlib\\n",
    "4. Creating and using a small untrained transformer model from a custom configuration\\n",
    "\\n",
    "These examples show how to use the core libraries included in this ML project template."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

    return notebook_content


def setup_jupyter():
    """Set up Jupyter notebook configuration if Jupyter support is enabled."""
    include_jupyter = "{{ cookiecutter.include_jupyter_support }}"

    if include_jupyter == "n":
        # Remove notebooks directory if Jupyter support is not enabled
        if os.path.exists("notebooks"):
            shutil.rmtree("notebooks")
    else:
        # Create an example notebook
        os.makedirs("notebooks", exist_ok=True)

        # Create the example notebook
        notebook_content = create_example_notebook()
        with open(os.path.join("notebooks", "ml_example.ipynb"), "w") as f:
            f.write(notebook_content)

        # Create a README file
        with open(os.path.join("notebooks", "README.md"), "w") as f:
            f.write("# Jupyter Notebooks\n\nThis directory contains Jupyter notebooks for interactive development and visualization.\n\n- `ml_example.ipynb`: Example notebook demonstrating numpy, torch, and transformers usage.\n")


def main():
    """Run all post-generation tasks."""
    print("Running post-generation tasks...")

    create_empty_dirs()
    create_gitignore()
    setup_jupyter()

    print("Post-generation tasks completed successfully!")


if __name__ == "__main__":
    main()
