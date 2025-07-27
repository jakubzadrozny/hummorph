import torch
from torch import nn


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.Tensor(3), requires_grad=True)

    def forward(self, rays, **kwargs):
        N = rays.shape[1]
        w = torch.randn((N, 3), device=rays.device)
        rgb = torch.sigmoid(w + self.param)
        return dict(rgb=rgb)
