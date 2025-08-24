"""Model utilities for machine learning projects."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class SimpleNN(nn.Module):
    """A simple neural network for demonstration purposes."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        """Initialize the model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output (e.g., number of classes)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_small_transformer(vocab_size=1000, hidden_size=64, num_layers=2, num_heads=2):
    """Create a small transformer model from a custom configuration.

    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Size of the hidden layers
        num_layers: Number of transformer layers
        num_heads: Number of attention heads

    Returns:
        A small BERT model
    """
    # Create a custom configuration
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=128
    )

    # Create the model
    model = BertModel(config)

    return model, config


def apply_model(model, input_data):
    """Apply a model to input data.

    Args:
        model: PyTorch model
        input_data: Input data for the model

    Returns:
        Model outputs
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
    return outputs
