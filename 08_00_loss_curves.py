from cProfile import label
import numpy as np
import test
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

    plt.legend(prop = {"size": 14})
    plt.show()

# plot_prediction()

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias=nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegression()

with torch.inference_mode():
    pred_y = model_0(x_test)

print(pred_y)
plot_prediction(predictions=pred_y)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr = 0.01)

torch.manual_seed(42)
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} Loss: {loss} TestLoss: {test_loss}")
        print(model_0.state_dict())

plot_prediction(predictions=test_pred)

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label = "Train loss")
plt.plot(epoch_count, test_loss_values, label = "Testing loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Trainining and test loss curves")
plt.legend()
plt.show()


    
