import torch
from torch import nn

weight = 0.7
bias = 0.3

start = 0 
end = 1
step = 0.02


X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
train_split = int (0.8 * len (X))

x_train, y_train = X[:train_split], y[:train_split]
x_test, y_test = X[train_split:], y[train_split:]

print(len(x_train), len(y_train), len(x_test), len(y_test))
