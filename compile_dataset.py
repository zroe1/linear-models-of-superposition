import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import pickle
from datetime import datetime

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for MNIST."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)  # Batch size x 784
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

class SimpleLinearClassifier(nn.Module):
    """Simple linear classifier for MNIST."""
    def __init__(self, input_size=784, num_classes=10):
        super(SimpleLinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)  # Batch size x 784
        logits = self.linear(x)
        return logits

def get_probabilities_with_temperature(logits, temperature=10.0):
    """Get softmax probabilities with temperature scaling."""
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=1)

def create_one_hot_labels(labels, num_classes=10):
    """Create hard one-hot vector labels."""
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

def compile_softmax_onehot_dataset(model, test_loader, temperature=10.0, save_path="softmax_onehot_dataset.pkl"):
    """
    Compile a dataset where X = softmax outputs and Y = one-hot labels from test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        temperature: Temperature for softmax scaling
        save_path: Path to save the compiled dataset
    
    Returns:
        X: Softmax outputs (N x 10)
        Y: One-hot labels (N x 10)
    """
    model.eval()
    
    all_softmax_outputs = []
    all_onehot_labels = []
    
    print(f"Compiling dataset from test set...")
    print(f"Using temperature: {temperature}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(test_loader)}")
                
            data, target = data.to(device), target.to(device)
            
            # Get model logits
            logits = model(data)
            
            # Apply temperature-scaled softmax
            softmax_outputs = get_probabilities_with_temperature(logits, temperature)
            
            # Create one-hot labels
            onehot_labels = create_one_hot_labels(target)
            
            # Move to CPU and store
            all_softmax_outputs.append(softmax_outputs.cpu())
            all_onehot_labels.append(onehot_labels.cpu())
    
    # Concatenate all batches
    X = torch.cat(all_softmax_outputs, dim=0)
    Y = torch.cat(all_onehot_labels, dim=0)
    
    print(f"\nDataset compiled:")
    print(f"  X shape (softmax outputs): {X.shape}")
    print(f"  Y shape (one-hot labels):  {Y.shape}")
    print(f"  Total samples: {X.shape[0]}")
    
    # Create metadata
    metadata = {
        'temperature': temperature,
        'num_samples': X.shape[0],
        'num_classes': X.shape[1],
        'created_at': datetime.now().isoformat(),
        'description': 'MNIST test set: X=temperature-scaled softmax outputs, Y=one-hot ground truth labels'
    }
    
    # Save the dataset
    dataset_dict = {
        'X': X,  # Softmax outputs
        'Y': Y,  # One-hot labels
        'metadata': metadata
    }
    
    # Save as pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    print(f"\n✓ Dataset saved to: {save_path}")
    
    # Also save as PyTorch tensors for easier loading
    torch_save_path = save_path.replace('.pkl', '.pt')
    torch.save(dataset_dict, torch_save_path)
    print(f"✓ Dataset also saved as PyTorch tensors to: {torch_save_path}")
    
    # Display some sample statistics
    print(f"\nDataset Statistics:")
    print(f"  Softmax outputs (X):")
    print(f"    Mean: {X.mean():.6f}")
    print(f"    Std:  {X.std():.6f}")
    print(f"    Min:  {X.min():.6f}")
    print(f"    Max:  {X.max():.6f}")
    
    print(f"  One-hot labels (Y):")
    print(f"    Sum per sample (should be 1.0): {Y.sum(dim=1).mean():.6f}")
    print(f"    Class distribution:")
    class_counts = Y.sum(dim=0)
    for i, count in enumerate(class_counts):
        print(f"      Class {i}: {int(count):,} samples")
    
    return X, Y, metadata

def load_compiled_dataset(load_path="softmax_onehot_dataset.pkl"):
    """
    Load a previously compiled dataset.
    
    Args:
        load_path: Path to the saved dataset
    
    Returns:
        X: Softmax outputs
        Y: One-hot labels
        metadata: Dataset metadata
    """
    try:
        # Try loading pickle first
        with open(load_path, 'rb') as f:
            dataset_dict = pickle.load(f)
    except FileNotFoundError:
        # Try PyTorch format
        torch_path = load_path.replace('.pkl', '.pt')
        try:
            dataset_dict = torch.load(torch_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find dataset at {load_path} or {torch_path}")
    
    return dataset_dict['X'], dataset_dict['Y'], dataset_dict['metadata']

def create_torch_dataset(X, Y):
    """Create a PyTorch TensorDataset from X and Y."""
    return TensorDataset(X, Y)

def main():
    # Configuration
    temperature = 20.0
    batch_size = 128
    save_path = "softmax_onehot_dataset.pkl"
    
    print("=" * 80)
    print("COMPILING SOFTMAX-ONEHOT DATASET FROM TEST SET")
    print("=" * 80)
    
    # Load the trained model
    print("Loading trained model...")
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    
    try:
        # Try to load the model saved by the main training script
        model.load_state_dict(torch.load('mnist_temperature_classifier.pth', map_location=device))
        print("✓ Loaded model: mnist_temperature_classifier.pth")
    except FileNotFoundError:
        try:
            # Fallback to the other model file
            model.load_state_dict(torch.load('trained_linear_model.pth', map_location=device))
            print("✓ Loaded model: trained_linear_model.pth")
        except FileNotFoundError:
            print("❌ No trained model found. Please run the training script first.")
            print("Expected files: 'mnist_temperature_classifier.pth' or 'trained_linear_model.pth'")
            return
    
    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load TEST data (not training data)
    print("Loading test data...")
    test_dataset = datasets.MNIST('data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print()
    
    # Compile the dataset
    X, Y, metadata = compile_softmax_onehot_dataset(
        model, test_loader, temperature=temperature, save_path=save_path
    )
    
    print("\n" + "=" * 80)
    print("DATASET COMPILATION COMPLETE")
    print("=" * 80)
    
    # Demonstrate how to load the dataset
    print("\nDemonstrating how to load the dataset:")
    print("```python")
    print("# Method 1: Using the load function")
    print("X, Y, metadata = load_compiled_dataset('softmax_onehot_dataset.pkl')")
    print()
    print("# Method 2: Direct pickle loading")
    print("import pickle")
    print("with open('softmax_onehot_dataset.pkl', 'rb') as f:")
    print("    data = pickle.load(f)")
    print("X, Y = data['X'], data['Y']")
    print()
    print("# Method 3: PyTorch tensor loading")
    print("data = torch.load('softmax_onehot_dataset.pt')")
    print("X, Y = data['X'], data['Y']")
    print()
    print("# Create PyTorch dataset for training")
    print("dataset = TensorDataset(X, Y)")
    print("dataloader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("```")

if __name__ == "__main__":
    main() 