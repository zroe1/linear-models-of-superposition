import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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

def temperature_scaled_cross_entropy(logits, targets, temperature=10.0):
    """Cross entropy loss with temperature scaling."""
    # Scale logits by temperature
    scaled_logits = logits / temperature
    # Apply standard cross entropy
    return F.cross_entropy(scaled_logits, targets)

def get_probabilities_with_temperature(logits, temperature=10.0):
    """Get softmax probabilities with temperature scaling."""
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=1)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, temperature=10.0):
    """Train the model with temperature-scaled softmax."""
    criterion = lambda logits, targets: temperature_scaled_cross_entropy(logits, targets, temperature)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy using temperature-scaled probabilities
            probs = get_probabilities_with_temperature(logits, temperature)
            _, predicted = torch.max(probs, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                
                # Use temperature-scaled probabilities for validation too
                probs = get_probabilities_with_temperature(logits, temperature)
                _, predicted = torch.max(probs, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
                
                val_loss += criterion(logits, target).item()
        
        # Calculate metrics
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, val_accuracies

def visualize_predictions(model, test_loader, temperature=10.0, num_samples=8):
    """Visualize some predictions with temperature-scaled probabilities."""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        logits = model(images)
        
        # Get both regular and temperature-scaled probabilities for comparison
        regular_probs = F.softmax(logits, dim=1)
        temp_probs = get_probabilities_with_temperature(logits, temperature)
        
        # Get predictions
        _, temp_predictions = torch.max(temp_probs, 1)
    
    # Plot results
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Plot image
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'True: {labels[i].item()}\nPred: {temp_predictions[i].item()}')
        axes[0, i].axis('off')
        
        # Plot probability distributions
        x = range(10)
        regular_prob = regular_probs[i].cpu().numpy()
        temp_prob = temp_probs[i].cpu().numpy()
        
        axes[1, i].bar(x, regular_prob, alpha=0.7, label=f'T=1.0', color='blue')
        axes[1, i].bar(x, temp_prob, alpha=0.7, label=f'T={temperature}', color='red')
        axes[1, i].set_xlabel('Class')
        axes[1, i].set_ylabel('Probability')
        axes[1, i].set_title('Probability Distribution')
        axes[1, i].legend()
        axes[1, i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mnist_temperature_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'mnist_temperature_predictions.png'")
    
    # Print some statistics about the probability distributions
    print(f"\nProbability Distribution Statistics (first sample):")
    print(f"Regular Softmax (T=1.0): Max={regular_probs[0].max():.3f}, Min={regular_probs[0].min():.3f}, Entropy={-torch.sum(regular_probs[0] * torch.log(regular_probs[0] + 1e-8)):.3f}")
    print(f"Temperature Softmax (T={temperature}): Max={temp_probs[0].max():.3f}, Min={temp_probs[0].min():.3f}, Entropy={-torch.sum(temp_probs[0] * torch.log(temp_probs[0] + 1e-8)):.3f}")

def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    temperature = 10.0  # Temperature for softmax scaling
    
    print(f"Training MNIST classifier with softmax temperature = {temperature}")
    print(f"This will make the probability distributions smoother and less spiky.\n")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load MNIST data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Split training into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    # Create model (you can switch between SimpleLinearClassifier and SimpleMLP)
    print("\nUsing SimpleMLP (Multi-Layer Perceptron)")
    model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10).to(device)
    
    # Alternative: Use simple linear classifier
    # print("\nUsing SimpleLinearClassifier")
    # model = SimpleLinearClassifier(input_size=784, num_classes=10).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print(f"\nStarting training with temperature = {temperature}...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        temperature=temperature
    )
    
    # Final test evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            probs = get_probabilities_with_temperature(logits, temperature)
            _, predicted = torch.max(probs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_accuracy = 100. * correct / total
    print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')
    
    # Visualize some predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, temperature=temperature)
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_temperature_classifier.pth')
    print(f"\nModel saved as 'mnist_temperature_classifier.pth'")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    main() 