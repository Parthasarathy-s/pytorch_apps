import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0 
end = 1
step = 0.02


X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
train_split = int (0.8 * len (X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

def plot_prediction(train_data = X_train,
                    train_label = y_train,
                    test_data = X_test,
                    test_label = y_test, 
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s = 4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s = 4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")

    plt.legend(prop = {"size": 14})
    plt.show()

plot_prediction(X_train, y_train, X_test, y_test)


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features= 1 , out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print('Model -> ', model_1, 'State Dict ->', model_1.state_dict())
