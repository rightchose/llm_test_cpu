import torch


input = torch.tensor([[1,2],[4,5]]).long()



embedding = torch.nn.Parameter(torch.randn(10,3))

print(embedding)

print("finish")