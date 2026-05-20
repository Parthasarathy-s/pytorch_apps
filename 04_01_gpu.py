import torch

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

print(torch.device("mps"))

#tensor default is cpu
tensor = torch.tensor([1,2,3])
print(tensor, tensor.device)

#create tensor with gpu
tensor = torch.tensor([1,2,3], device="mps")

#tensor not on gpu
print(tensor, tensor.device)
