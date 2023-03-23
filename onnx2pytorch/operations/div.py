import torch
from torch import nn


class Div(nn.Module):
    def forward(self, input, other):
        res_type = torch.float32
        true_quotient = torch.div(input, other)
        if res_type.is_floating_point:
            res = true_quotient
        else:
            res = torch.floor(true_quotient).to(res_type)
        return res
