import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

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

def format_vector(vector, decimals=3):
    """Format a vector for readable display."""
    formatted = [f"{val:.{decimals}f}" for val in vector]
    return "[" + ", ".join(formatted) + "]"

def inspect_model_outputs(model, data_loader, temperature=10.0, num_samples=100):
    """Inspect model outputs vs one-hot labels for training data."""
    model.eval()
    
    print("=" * 120)
    print(f"MODEL OUTPUT INSPECTION - {num_samples} SAMPLES")
    print(f"Temperature: {temperature}")
    print("=" * 120)
    print()
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if samples_collected >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            logits = model(data)
            
            # Get temperature-scaled probabilities
            probs = get_probabilities_with_temperature(logits, temperature)
            
            # Create one-hot labels
            one_hot_labels = create_one_hot_labels(target)
            
            # Process each sample in the batch
            batch_size = data.size(0)
            for i in range(min(batch_size, num_samples - samples_collected)):
                sample_num = samples_collected + 1
                true_label = target[i].item()
                model_output = probs[i].cpu().numpy()
                one_hot_label = one_hot_labels[i].cpu().numpy()
                
                # Get predicted class
                predicted_class = torch.argmax(probs[i]).item()
                prediction_confidence = model_output[predicted_class]
                
                print(f"SAMPLE {sample_num:3d}")
                print(f"  True Label: {true_label}")
                print(f"  Predicted:  {predicted_class} (confidence: {prediction_confidence:.3f})")
                print(f"  Correct:    {'✓' if predicted_class == true_label else '✗'}")
                print()
                
                print(f"  One-Hot Label:   {format_vector(one_hot_label, decimals=0)}")
                print(f"  Model Output:    {format_vector(model_output, decimals=3)}")
                print()
                
                # Show class-wise breakdown
                print("  Class-wise Breakdown:")
                print("  " + "-" * 60)
                print("  Class | One-Hot | Model Output | Difference")
                print("  " + "-" * 60)
                
                for class_idx in range(10):
                    one_hot_val = one_hot_label[class_idx]
                    model_val = model_output[class_idx]
                    diff = abs(one_hot_val - model_val)
                    
                    marker = " ← TRUE" if class_idx == true_label else ""
                    marker += " ← PRED" if class_idx == predicted_class and class_idx != true_label else ""
                    
                    print(f"    {class_idx:3d}   |   {one_hot_val:3.0f}   |    {model_val:6.3f}    |   {diff:6.3f}{marker}")
                
                print("  " + "-" * 60)
                print()
                
                # Calculate and display some metrics for this sample
                mse = np.mean((one_hot_label - model_output) ** 2)
                l1_distance = np.sum(np.abs(one_hot_label - model_output))
                entropy = -np.sum(model_output * np.log(model_output + 1e-8))
                
                print(f"  Sample Metrics:")
                print(f"    Mean Squared Error:  {mse:.6f}")
                print(f"    L1 Distance:         {l1_distance:.6f}")
                print(f"    Output Entropy:      {entropy:.6f}")
                print()
                print("=" * 120)
                print()
                
            samples_collected += min(batch_size, num_samples - samples_collected)
    
    print(f"Inspection complete. Analyzed {samples_collected} samples.")

def main():
    # Load the trained model
    temperature = 20.0
    
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
    
    # Load training data
    print("Loading training data...")
    train_dataset = datasets.MNIST('data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Training dataset size: {len(train_dataset)} samples")
    print()
    
    # Inspect model outputs
    inspect_model_outputs(model, train_loader, temperature=temperature, num_samples=100)

if __name__ == "__main__":
    main() 