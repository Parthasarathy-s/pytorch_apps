import torch
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)

Y = weight * X + bias


train_split = int(len(X) * 0.8)

X_train, y_train = X[:train_split], Y[:train_split]
X_test, y_test = X[train_split:], Y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

def plot_predictions(train_data = X_train , 
                     train_label = y_train, 
                     test_data = X_test, 
                     test_label = y_test, 
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_label, c = "b", s = 4, label = "Training data")

    plt.scatter(test_data, test_label, c = "g", s = 4, label = "Testing data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")

    plt.legend(prop = {"size": 14})
    plt.show()

plot_predictions()
