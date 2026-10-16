
import torch
import torch.nn as nn

# Sample training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

# 1. Model: z = wx + b
model = nn.Linear(1, 1)

# 2. Loss function
criterion = nn.BCEWithLogitsLoss()

# 3. Gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. Training loop
for epoch in range(1000):
    z = model(X)             # Linear score
    loss = criterion(z, y)   # Log-loss from logits

    optimizer.zero_grad()
    loss.backward()          # Calculate gradients
    optimizer.step()         # Update weights and bias

# Predict probabilities
with torch.no_grad():
    probabilities = torch.sigmoid(model(X))
    print(probabilities)
