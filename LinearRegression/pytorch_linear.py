import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Our linear regression model By passing nn.Module we can call methods from the nn.Module class which is the base
# class for all neural networks with pytorch
class Linear(nn.Module):
    def __init__(self):
        # Initializes the parent class within the child class
        super(Linear, self).__init__()
        # Create linear layer that takes 2 input features and returns 1 output
        self.linear = nn.Linear(2, 1)

    # Called when data is passed through network
    def forward(self, x):
        # Passes x through our linear layer
        x = self.linear(x)
        # Calls activation function relu
        x = F.relu(x)
        # Returns x
        return x


model = Linear()

# Inputs
# Make them a tensor
x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y_train = torch.tensor([[1.0], [2.0], [3.0]])

# What loss function to use
criterion = nn.MSELoss()
# What optimizer to use
# Responsible for updating model's weights and biases
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
# 20 epochs
for i in range(20):
    # Forwarding data
    output = model(x_train)
    # Calculating loss
    loss = criterion(output, y_train)
    # Optimization
    # Reset gradients
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Update parameters based of gradients
    optimizer.step()
    print(f'Epoch [{i + 1}/20], Loss:', loss)

print("Training Complete")

print(model(torch.tensor([[6.0, 7.0]])))

# Tensor: In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of
# algebraic objects
# Gradient: A rate of inclination; a slope.
