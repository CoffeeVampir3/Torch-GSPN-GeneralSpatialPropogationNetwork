import torch
import torch.nn as nn
import torch.nn.functional as F

#https://arxiv.org/abs/2301.00808

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: input features with shape [N,H,W,C]

        # Equation (1): Global feature aggregation using L2 norm
        gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)  # G(X)i = ||Xi||

        # Equation (2): Feature normalization - N(||Xi||) = ||Xi|| / (Σ||Xj||)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)

        # Equation (3): Feature calibration - Xi = Xi * N(G(X)i)
        return self.gamma * (x * nx) + self.beta + x