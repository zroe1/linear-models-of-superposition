import torch
import torch.nn as nn
import torch.nn.functional as F
from plots import create_enhanced_phase_diagram
from compile_dataset import load_compiled_dataset

# order of priority: gpu -> mps -> cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

NUM_CLASSES = 10
NUM_FEATURES = 2
IMPORTANCE = (0.85 ** torch.arange(0, NUM_CLASSES)).to(device)

def smallest_angle_between_weights(model):
    """
    Compute the smallest angle between any pair of weight vectors in the model.
    
    Args:
        model: ToyModelLinear model with weights of shape (NUM_FEATURES, NUM_CLASSES)
        
    Returns:
        float: Smallest angle in degrees between any pair of weight vectors
    """
    import math
    
    weights = model.weights.detach().cpu()  # Shape: (NUM_FEATURES, NUM_CLASSES)
    num_classes = weights.shape[1]
    
    if num_classes < 2:
        return 0.0
    
    min_angle = float('inf')
    
    # Compute angle between all pairs of weight vectors
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            vec1 = weights[:, i]  # Shape: (NUM_FEATURES,)
            vec2 = weights[:, j]  # Shape: (NUM_FEATURES,)
            
            # Compute norms
            norm1 = torch.norm(vec1)
            norm2 = torch.norm(vec2)
            
            # Skip if either vector is zero
            if norm1 == 0 or norm2 == 0:
                continue
                
            # Compute cosine of angle
            cos_angle = torch.dot(vec1, vec2) / (norm1 * norm2)
            
            # Clamp to avoid numerical errors
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            
            # Compute angle in radians, then convert to degrees
            angle_radians = torch.acos(cos_angle).item()
            angle_degrees = math.degrees(angle_radians)
            
            # Update minimum angle
            min_angle = min(min_angle, angle_degrees)
    
    return min_angle if min_angle != float('inf') else 0.0

def is_superposition(model, epsilon=None):
    """
    Determine if the model is in a superposition regime by checking if the smallest
    angle between weight vectors is close to the expected angle for evenly distributed vectors.
    
    Args:
        model: ToyModelLinear model with weights of shape (NUM_FEATURES, NUM_CLASSES)
        epsilon: Tolerance for angle comparison in degrees (default: 5.7)
        
    Returns:
        bool: True if the model appears to be in superposition, False otherwise
    """
    # Get number of classes from the model's weight shape
    num_classes = model.weights.shape[1]
    
    if num_classes < 2:
        return False
    
    if epsilon is None:
        epsilon = (360.0 / num_classes) *0.2

    # Expected angle between adjacent vectors in superposition (360/NUM_CLASSES degrees)
    expected_angle_degrees = 360.0 / num_classes
    
    # Get the smallest angle between any pair of weight vectors (in degrees)
    smallest_angle = smallest_angle_between_weights(model)
    
    # Check if the smallest angle is within epsilon of the expected angle
    return abs(smallest_angle - expected_angle_degrees) <= epsilon

class ToyModelLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(NUM_FEATURES, NUM_CLASSES) , requires_grad=True)
        self.bias = nn.Parameter(torch.randn(NUM_CLASSES, 1) , requires_grad=True)
        
    def forward(self, x): # x is NUM_FEATURES * 1
        hidden = self.weights @ x
        final = self.weights.T @ hidden
        final += self.bias
        return final

def mse_loss(output, target):
    """
    Standard MSE loss for reconstruction.
    """
    return torch.mean((output - target)**2)

def evaluate_reconstruction_quality(model, X_data):
    """
    Evaluate the model's reconstruction quality using MSE.
    
    Args:
        model: The trained model to evaluate
        X_data: Input data (softmax outputs to reconstruct)
        
    Returns:
        float: Average reconstruction MSE loss
    """
    model.eval()
    total_loss = 0.0
    num_samples = X_data.shape[0]
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get single sample
            x_sample = X_data[i].unsqueeze(1).to(device)  # Shape: (NUM_FEATURES, 1)
            
            # Get model reconstruction
            output = model(x_sample)  # Shape: (NUM_FEATURES, 1)
            
            # Calculate MSE loss
            loss = mse_loss(output, x_sample)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_samples
    return avg_loss

def train_linear(model, epochs, total_batchs, batch_size, optimizer, X_train):
    model.train()
    loss_total = 0
    
    # Store initial learning rate for cosine decay
    initial_lr = optimizer.param_groups[0]['lr']
    import math
    
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Apply cosine decay: lr = initial_lr * 0.5 * (1 + cos(π * epoch / total_epochs))
        cosine_lr = initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosine_lr
            
        for i in range(total_batchs):
            # Sample random batch from the dataset
            batch_indices = torch.randint(0, num_samples, (batch_size,))
            x_batch = X_train[batch_indices].to(device)  # Shape: (batch_size, NUM_FEATURES)

            loss = torch.tensor([0.0]).to(device)
            for b in range(batch_size):
                # Convert to the format expected by the model (NUM_FEATURES, 1)
                x_sample = x_batch[b].unsqueeze(1)  # Shape: (NUM_FEATURES, 1)
                
                # Model tries to reconstruct the input
                pred = model(x_sample).to(device)
                
                # MSE loss between reconstruction and original input
                loss += mse_loss(pred, x_sample)
            
            loss_total += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if is_superposition(model):
            print("Superposition detected")
            break

        # Evaluate reconstruction quality (sample 1000 for efficiency)
        sample_size = min(1000, num_samples)
        sample_indices = torch.randperm(num_samples)[:sample_size]
        reconstruction_mse = evaluate_reconstruction_quality(model, X_train[sample_indices])
        avg_loss = loss_total / (total_batchs * batch_size)
        print(f"EPOCH: {epoch + 1} --> loss: {avg_loss:.6f}, reconstruction_mse: {reconstruction_mse:.6f}, lr: {cosine_lr:.6f}")
        
        # Debug: print some reconstructions on first few epochs
        if epoch < 3:
            model.eval()
            with torch.no_grad():
                test_input = X_train[0].unsqueeze(1).to(device)  # First sample
                test_output = model(test_input)
                print(f"  Debug - Input:         {test_input.squeeze()[:5]}")
                print(f"  Debug - Reconstruction: {test_output.squeeze()[:5]}")
                print(f"  Debug - MSE:           {mse_loss(test_output, test_input).item():.6f}")
            model.train()
        
        loss_total = 0

if __name__ == "__main__":

    NUM_EPOCHS = 50
    BATCHS_PER_EPOCH = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-2

    print("="*60)
    print("Training Linear Autoencoder on Softmax Outputs")
    print("="*60)
    
    # Load the compiled dataset
    print("Loading compiled dataset...")
    try:
        X, Y, metadata = load_compiled_dataset('softmax_onehot_dataset.pkl')
        print(f"✓ Loaded dataset: {X.shape[0]} samples")
        print(f"  Temperature used: {metadata['temperature']}")
        print(f"  Input shape (softmax): {X.shape}")
        print("  Task: Reconstruct softmax inputs (autoencoder)")
    except FileNotFoundError:
        print("❌ Dataset not found. Please run compile_dataset.py first.")
        exit(1)
    
    # Use 80% for training, 20% for final evaluation
    num_samples = X.shape[0]
    num_train = int(0.8 * num_samples)
    
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    print(f"Training samples: {num_train}, Test samples: {len(test_indices)}")
    
    model_linear = ToyModelLinear().to(device)
    optimizer_linear = torch.optim.Adam(model_linear.parameters(), lr=LEARNING_RATE)

    train_linear(model_linear, NUM_EPOCHS, BATCHS_PER_EPOCH, BATCH_SIZE, optimizer_linear, X_train)
    print(f"Linear model smallest angle: {smallest_angle_between_weights(model_linear)}")
    
    # Evaluate linear model reconstruction quality on test set
    reconstruction_mse = evaluate_reconstruction_quality(model_linear, X_test)
    print(f"Linear Model Test Reconstruction MSE: {reconstruction_mse:.6f}")
    print(f"Is in superposition: {is_superposition(model_linear)}")

    # Save the trained linear model
    torch.save(model_linear.state_dict(), 'trained_linear_autoencoder.pth')
    print("Linear autoencoder saved as 'trained_linear_autoencoder.pth'")

    # Note: Phase diagram skipped - designed for 2D features, but we have 10D features
    phase_data = create_enhanced_phase_diagram(model_linear.weights.T.detach().cpu(), model_linear.weights.T.detach().cpu(), 
                                           model_linear.bias.detach().cpu(), 'cpu')

    print(f"\nSummary:")
    print(f"Linear Autoencoder - Test Reconstruction MSE: {reconstruction_mse:.6f}, Angle: {smallest_angle_between_weights(model_linear):.2f}°")
    print(f"Dataset: {num_samples} total samples, Temperature: {metadata['temperature']}")
    
    # Show some example reconstructions
    print(f"\nExample Reconstructions (first 5 test samples):")
    model_linear.eval()
    with torch.no_grad():
        for i in range(min(5, len(X_test))):
            input_sample = X_test[i].unsqueeze(1).to(device)
            reconstruction = model_linear(input_sample)
            mse = mse_loss(reconstruction, input_sample).item()
            print(f"  Sample {i+1}:")
            print(f"    Input:   {input_sample.squeeze().cpu().numpy()}")
            print(f"    Recon:   {reconstruction.squeeze().cpu().numpy()}")
            print(f"    MSE:     {mse:.6f}") 