import torch
import torch.nn as nn
import torch.nn.functional as F
from plots import create_enhanced_phase_diagram

# order of priority: gpu -> mps -> cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ToyModelLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2, 5), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(5, 1), requires_grad=True)
        
    def forward(self, x): # x is 5 * 1
        hidden = self.weights @ x
        final = self.weights.T @ hidden
        final += self.bias
        return final
    
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
    return (logits[target] - input_t)**2 + logits[i_neq_t] * 0.01
    
def train_linear(model, epochs, total_batchs, batch_size, optimizer):
    model.train()
    loss_total = 0
    for epoch in range(epochs):
        for i in range(total_batchs):
            x = torch.rand(batch_size, 5, 1).to(device)

            targets =  torch.randint(0, 5, (batch_size,)).to(device)

            sparsity_tensor = torch.zeros_like(x).to(device)
            batch_indices = torch.arange(batch_size).to(device)
            sparsity_tensor[batch_indices, targets, 0] = 1

            
            x = (x*sparsity_tensor).to(device)
            pred = model(x)

            loss = torch.tensor([0.0]).to(device)
            for b in range(batch_size):
                p = pred[b].to(device)
                target = targets[b].to(device)
                # loss += f6(p, target)
                loss += zephy_loss1(p, target, x[b][target])
            loss_total += loss.item()
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("EPHOCH:", epoch + 1, "--> loss:", loss_total / (total_batchs * batch_size))
        loss_total = 0

if __name__ == "__main__":

    NUM_EPOCHS = 5
    BATCHS_PER_EPOCH =400
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-3


    model = ToyModelLinear().to(device)

    SPARSITY = 0.95
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # loss_func = ImporanceWeightedMSE()

    train_linear(model, NUM_EPOCHS, BATCHS_PER_EPOCH, BATCH_SIZE, optimizer)


    phase_data = create_enhanced_phase_diagram(model.weights.T.detach().cpu(), model.weights.T.detach().cpu(), 
                                           model.bias.detach().cpu(), 'cpu')