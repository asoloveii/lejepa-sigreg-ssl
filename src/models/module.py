import torch, torch.nn as nn


class Derf(nn.Module):
    """Dynamic erf layer with learnable shift and scale operations"""
    
    def __init__(self, d_model, init_shift, init_scale):
        super().__init__()

        self.shift = nn.Parameter(torch.ones(1)*init_shift, requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1)*init_scale, requires_grad=True)

        self.gamma = nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return self.gamma * torch.erf(self.shift * x + self.scale) + self.beta
