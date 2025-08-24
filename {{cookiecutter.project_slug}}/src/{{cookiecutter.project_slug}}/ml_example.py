"""Example script demonstrating basic ML operations with numpy, torch, and matplotlib."""

import numpy as np
import torch
import matplotlib.pyplot as plt

def numpy_calculations():
    """Perform some basic calculations using numpy."""
    print("Numpy Calculations:")
    
    # Create a random matrix
    matrix = np.random.rand(5, 5)
    print(f"Random matrix:\n{matrix}")
    
    # Matrix operations
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print(f"\nEigenvalues:\n{eigenvalues}")
    
    # Matrix multiplication
    result = np.matmul(matrix, matrix.T)
    print(f"\nMatrix multiplied by its transpose:\n{result}")
    
    # Statistical operations
    print(f"\nMean of each column: {np.mean(matrix, axis=0)}")
    print(f"Standard deviation of each column: {np.std(matrix, axis=0)}")
    
    return matrix

def torch_calculations():
    """Perform some basic calculations using PyTorch."""
    print("\nPyTorch Calculations:")
    
    # Create a random tensor
    tensor = torch.rand(5, 5)
    print(f"Random tensor:\n{tensor}")
    
    # Tensor operations
    result = torch.matmul(tensor, tensor.t())
    print(f"\nTensor multiplied by its transpose:\n{result}")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_on_device = tensor.to(device)
    print(f"\nTensor moved to {device}")
    
    # Neural network operations
    # Create a simple linear layer
    linear = torch.nn.Linear(5, 3).to(device)
    output = linear(tensor_on_device)
    print(f"\nOutput of linear layer:\n{output}")
    
    # Activation function
    activated = torch.nn.functional.relu(output)
    print(f"\nAfter ReLU activation:\n{activated}")
    
    return tensor, output

def plot_data(numpy_data, torch_data):
    """Create and display plots using matplotlib."""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Heatmap of numpy matrix
    plt.subplot(2, 2, 1)
    plt.imshow(numpy_data, cmap='viridis')
    plt.colorbar()
    plt.title('Numpy Matrix Heatmap')
    
    # Plot 2: Line plot of numpy data
    plt.subplot(2, 2, 2)
    for i in range(numpy_data.shape[1]):
        plt.plot(numpy_data[:, i], label=f'Column {i+1}')
    plt.title('Numpy Matrix Columns')
    plt.legend()
    
    # Plot 3: Heatmap of torch tensor
    plt.subplot(2, 2, 3)
    plt.imshow(torch_data.detach().cpu().numpy(), cmap='plasma')
    plt.colorbar()
    plt.title('PyTorch Output Heatmap')
    
    # Plot 4: Bar chart of torch data mean values
    plt.subplot(2, 2, 4)
    means = torch.mean(torch_data, dim=0).detach().cpu().numpy()
    plt.bar(range(len(means)), means)
    plt.title('Mean Values of PyTorch Output')
    plt.xlabel('Output Dimension')
    plt.ylabel('Mean Value')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figures/ml_example_plots.png')
    print("\nPlots saved to 'figures/ml_example_plots.png'")
    
    # Show the plot
    plt.show()

def main():
    """Run the example."""
    print("ML Example: Numpy, PyTorch, and Matplotlib")
    print("=" * 50)
    
    # Create directory for figures
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Run numpy calculations
    numpy_data = numpy_calculations()
    
    # Run torch calculations
    torch_data, torch_output = torch_calculations()
    
    # Create and display plots
    plot_data(numpy_data, torch_output)
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()