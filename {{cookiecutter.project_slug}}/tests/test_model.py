"""Tests for the model module."""

import torch
import pytest
from src.model import SimpleNN


def test_simple_nn_forward():
    """Test the forward pass of the SimpleNN model."""
    # Create a model
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    model = SimpleNN(input_dim, hidden_dim, output_dim)

    # Create a batch of data
    batch_size = 3
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, output_dim)


def test_simple_nn_parameters():
    """Test the parameters of the SimpleNN model."""
    # Create a model
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    model = SimpleNN(input_dim, hidden_dim, output_dim)

    # Check that parameters exist and have the right shapes
    assert hasattr(model, 'fc1')
    assert model.fc1.weight.shape == (hidden_dim, input_dim)

    assert hasattr(model, 'fc2')
    assert model.fc2.weight.shape == (output_dim, hidden_dim)


def test_simple_nn_training():
    """Test that the SimpleNN model can be trained."""
    # Create a model
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    model = SimpleNN(input_dim, hidden_dim, output_dim)

    # Create a batch of data
    batch_size = 3
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initial forward pass
    output = model(x)
    initial_loss = criterion(output, y)

    # Train for a few steps
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Check that the loss decreased
    final_loss = criterion(model(x), y)
    assert final_loss < initial_loss
