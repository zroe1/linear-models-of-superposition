import torch
import torch.nn as nn
import torch.nn.functional as F
from plots import create_enhanced_phase_diagram

# order of priority: gpu -> mps -> cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

NUM_CLASSES = 100
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

class ImporanceWeightedMSE(nn.Module):
    def __init__(self):
        super(ImporanceWeightedMSE, self).__init__()

    def forward(self, predictions, targets, importance):
        predictions = F.relu(predictions)
        sub_total = ((predictions - targets)**2).flatten()
        return sum(sub_total * importance)

class ImporanceWeightedMSEBatch(nn.Module):
    """Improved version that handles batches properly like in reference implementation."""
    def __init__(self):
        super(ImporanceWeightedMSEBatch, self).__init__()

    def forward(self, predictions, targets, importance):
        # Handle batch dimension properly
        sub_total = ((predictions - targets)**2).sum(0).flatten()
        return sum(sub_total * importance)

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

class ToyModelReLU(nn.Module):
    """ReLU model based on the reference implementation for 90% sparsity training."""
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(NUM_FEATURES, NUM_CLASSES), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(NUM_CLASSES, 1), requires_grad=True)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x): # x is NUM_CLASSES * 1
        hidden = self.weights @ x
        final = self.weights.T @ hidden
        final += self.bias
        return self.ReLU(final)
    
def f6(logits, target):
    i_neq_t = torch.argmax(logits)
    if i_neq_t == target:
        masked_logits = logits.clone()
        masked_logits[target] = float('-inf')
        i_neq_t = torch.argmax(masked_logits)
    return F.relu(logits[i_neq_t] - logits[target])

def zephy_loss1(logits, target, input_t):
    i_neq_t = torch.argmax(logits)
    if i_neq_t == target:
        masked_logits = logits.clone()
        masked_logits[target] = float('-inf')
        i_neq_t = torch.argmax(masked_logits)
    # return (logits[target] - input_t)**2 + logits[i_neq_t] * 0.0001
    return (logits[target] - input_t)**2 + logits[i_neq_t] * 0

# def zephy_loss1(logits, target, input_t):
#     i_neq_t = torch.argmax(logits)
#     if i_neq_t == target:
#         masked_logits = logits.clone()
#         masked_logits[target] = float('-inf')
#         i_neq_t = torch.argmax(masked_logits)
#     return (logits[target] - input_t)**2 + logits[i_neq_t] * 0.01
    
def relu_mse_loss(logits, targets):
    return torch.mean((F.relu(logits) - targets)**2)

def evaluate_accuracy(model):
    """
    Evaluate the model's accuracy on one-hot vectors.
    
    For each class, create a one-hot vector and check if the model's
    top prediction matches the correct class.
    
    Args:
        model: The trained model to evaluate
        
    Returns:
        float: Accuracy as a percentage (0-100)
    """
    model.eval()
    correct_predictions = 0
    total_predictions = NUM_CLASSES
    
    with torch.no_grad():
        for class_idx in range(NUM_CLASSES):
            # Create one-hot vector for this class
            one_hot = torch.zeros(NUM_CLASSES, 1).to(device)
            one_hot[class_idx, 0] = 1.0
            
            # Get model prediction
            output = model(one_hot.unsqueeze(0))  # Add batch dimension
            predicted_class = torch.argmax(output[0]).item()
            
            # Check if prediction matches the true class
            if predicted_class == class_idx:
                correct_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100.0
    return accuracy

def train_linear(model, epochs, total_batchs, batch_size, optimizer, sparsity):
    model.train()
    loss_total = 0
    
    # Store initial learning rate for cosine decay
    initial_lr = optimizer.param_groups[0]['lr']
    import math

    for epoch in range(epochs):
        # Apply cosine decay: lr = initial_lr * 0.5 * (1 + cos(π * epoch / total_epochs))
        cosine_lr = initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosine_lr
            
        for i in range(total_batchs):
            x = torch.rand(batch_size, NUM_CLASSES, 1).to(device)

            targets =  torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)

            target_tensor = torch.zeros_like(x).to(device)
            batch_indices = torch.arange(batch_size).to(device)

            # create tensor of 1s and 0s with probability sparsity
            sparsity_tensor = torch.bernoulli(torch.full((batch_size, NUM_CLASSES, 1), sparsity)).to(device)
            sparsity_tensor = sparsity_tensor * 0.1
            # sparsity_tensor = torch.zeros_like(x).to(device)
            sparsity_tensor[batch_indices, targets, 0] = 1
            assert x.shape == sparsity_tensor.shape
            
            x = (x*sparsity_tensor).to(device)
            pred = model(x)

            loss = torch.tensor([0.0]).to(device)
            for b in range(batch_size):
                p = pred[b].to(device)
                target = targets[b].to(device)
                loss += zephy_loss1(p, target, x[b][target])
            loss_total += loss.item()

            loss_total += loss.item()
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if is_superposition(model):
            print("Superposition detected")
            break

        # Evaluate and print accuracy at each epoch
        accuracy = evaluate_accuracy(model)
        print(f"EPOCH: {epoch + 1} --> loss: {loss_total / (total_batchs * batch_size):.6f}, accuracy: {accuracy:.2f}%, lr: {cosine_lr:.6f}")
        loss_total = 0

def train_relu(model, epochs, total_batchs, batch_size, loss_fn, optimizer, importance, sparsity):
    """Train ReLU model with specified sparsity level based on reference implementation."""
    probability = 1 - sparsity
    model.train()
    loss_total = 0
    
    for epoch in range(epochs):
        for i in range(total_batchs):
            # Generate sparse data using bernoulli distribution like in reference
            sparsity_tensor = torch.bernoulli(torch.full((NUM_CLASSES, 1), probability)).to(device)
            x = torch.rand(batch_size, NUM_CLASSES, 1).to(device)
            x = (x * sparsity_tensor).to(device)
            
            pred = model(x)
            loss = loss_fn(pred, x, importance)
            loss_total += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if is_superposition(model):
            print("Superposition detected")
            break

        print("EPOCH:", epoch + 1, "--> loss:", loss_total / (total_batchs * batch_size))
        loss_total = 0

if __name__ == "__main__":

    # NUM_EPOCHS = 200
    NUM_EPOCHS = 200
    BATCHS_PER_EPOCH =50
    # BATCH_SIZE = 256
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-2


    model = ToyModelLinear().to(device)

    SPARSITY = 0.95
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # loss_func = ImporanceWeightedMSE()

    train_linear(model, NUM_EPOCHS, BATCHS_PER_EPOCH, BATCH_SIZE, optimizer, SPARSITY)
    print(smallest_angle_between_weights(model))
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(model)
    print(f"Linear Model Accuracy: {accuracy:.2f}%")
    print(f"Is in superposition: {is_superposition(model)}")
 
    # Save the trained model
    torch.save(model.state_dict(), 'trained_linear_model.pth')
    print("Model saved as 'trained_linear_model.pth'")

    phase_data = create_enhanced_phase_diagram(model.weights.T.detach().cpu(), model.weights.T.detach().cpu(), 
                                           model.bias.detach().cpu(), 'cpu')

    
    
    # print("\n" + "="*50)
    # print("Training 90% Sparsity ReLU Model")
    # print("="*50)
    
    # SPARSITY = 0.95
    # model_relu = ToyModelReLU().to(device)
    # optimizer_relu = torch.optim.SGD(model_relu.parameters(), lr=LEARNING_RATE)
    # loss_func_relu = ImporanceWeightedMSEBatch()
    
    # train_relu(model_relu, NUM_EPOCHS, BATCHS_PER_EPOCH, BATCH_SIZE, loss_func_relu, optimizer_relu, IMPORTANCE, SPARSITY)
    # print("Smallest angle between weights (ReLU model):", smallest_angle_between_weights(model_relu))
    # print("Is in superposition (ReLU model):", is_superposition(model_relu))
    
    # # Evaluate ReLU model accuracy
    # accuracy_relu = evaluate_accuracy(model_relu)
    # print(f"ReLU Model Accuracy: {accuracy_relu:.2f}%")
    
    # print("\nReLU model weights:")
    # print(model_relu.weights.T.detach().cpu())

    # phase_data = create_enhanced_phase_diagram(model_relu.weights.T.detach().cpu(), model_relu.weights.T.detach().cpu(), 
    #                                        model_relu.bias.detach().cpu(), 'cpu')