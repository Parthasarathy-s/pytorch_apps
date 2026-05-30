from cProfile import label

from torch import nn
import torch
from numpy import gradient
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0 
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(1)
print(len(X))

Y = weight * X + bias

train_split = int(len(X) * 0.8)

x_train, y_train = X[:train_split], Y[:train_split]
x_test, y_test = X[train_split:], Y[train_split:]

print(len(x_train), len(y_train), len(x_test), len(y_test))


def plot_prediction(train_data = x_train,
                    train_label = y_train,
                    test_data = x_test,
                    test_label = y_test, 
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s = 4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s = 4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")

    plt.legend(prep={"size": 14})
    plt.show()

plot_prediction()
