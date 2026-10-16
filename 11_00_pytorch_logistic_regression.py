
import torch
import torch.nn as nn

# Sample training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

# 1. Model: z = wx + b
model = nn.Linear(1, 1)

# 2. Loss function
criterion = nn.BCEWithLogitsLoss()

# 3. Gradient descent optimizer - Stochastic Gradient Descent.
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

"""
Hours studied (X) | Probability of passing | Prediction (threshold = 0.5)
1 | 3.47% | Fail (0)
2 | 28.32% | Fail (0)
3 | 81.28% | Pass (1)
4 | 97.95% | Pass (1)


Step 1 — Forward Pass Code: z = model(X) Mathematics: z = wx + b Explanation: Calculates the model's score using the input, weight, and bias.

Step 2 — Calculate Loss Code: loss = criterion(z, y) Mathematics: L = -[y·ln(ŷ) + (1-y)·ln(1-ŷ)] Explanation: Measures how far the predicted probabilities are from the actual labels. In PyTorch, BCEWithLogitsLoss combines sigmoid and binary cross-entropy internally.

Step 3 — Calculate Gradients Code: loss.backward() Mathematics: ∂L/∂w = (ŷ - y)·x ∂L/∂b = (ŷ - y) Explanation: PyTorch automatically calculates how the loss changes with respect to the weight and bias.

Step 4 — Gradient Descent Update Code: optimizer.step() Mathematics: w = w - α·∂L/∂w b = b - α·∂L/∂b Explanation: Updates the weight and bias using the calculated gradients to reduce the loss.

Step 5 — Clear Previous Gradients Code: optimizer.zero_grad() Explanation: Resets the gradients from the previous iteration before calculating new ones. PyTorch accumulates gradients by default.

Remember:
Forward pass = Calculate predictions.
Loss = Measure prediction error.
Backpropagation = Calculate gradients.
Gradient descent = Update parameters.
Zero gradients = Prepare for the next iteration.

"""
