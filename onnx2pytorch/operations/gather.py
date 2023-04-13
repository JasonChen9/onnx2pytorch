import torch
from torch import nn


class Gather(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        selection = self.selection + [indices.to(torch.int64)]
        device = data.device
        return data.cpu().__getitem__(selection).to(device)
