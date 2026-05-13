import torch

tensor = torch.arange(0, 100, 10)
print(tensor)


print(torch.sum(tensor), " - ", tensor.sum())

print(torch.max(tensor), " - ", tensor.max())

print(torch.mean(tensor.type(torch.float32)), " - ", tensor.type(torch.float32).mean())
