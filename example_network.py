import torch
import torch.nn as nn
from model.gspn import GSPNLayer

class GSPNBlock(nn.Module):
    def __init__(self, dim, is_global=True, group_size=2):
        super(GSPNBlock, self).__init__()
        self.gspn = GSPNLayer(dim, is_global=is_global, group_size=group_size)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Input shape: B C H W
        B, C, H, W = x.shape
        x = x + self.gspn(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x + self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x

class GSPNNetwork(nn.Module):
    def __init__(self, in_channels=3, dims=[96, 192, 384, 768], depths=[2, 2, 7, 2], group_size=2):
        super(GSPNNetwork, self).__init__()
        # project input into feature space
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)

        self.levels = nn.ModuleList()
        for i in range(len(dims)):
            level = nn.Sequential(
                *[GSPNBlock(dims[i], is_global=(i >= 2), group_size=group_size) for _ in range(depths[i])]
            )
            self.levels.append(level)

            # downsample between levels (except the last level)
            if i < len(dims) - 1:
                self.levels.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        # Head -- Rip this off to use network for something actually useful
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # [B, C]
            nn.Linear(dims[-1], 1000)  # 1000 classes for ImageNet
        )

    def forward(self, x):
        # Input shape: B C H W
        x = self.stem(x)
        for level in self.levels:
            x = level(x)
            
        x = self.head(x)  # Classification head
        return x

if __name__ == "__main__":
    model = GSPNNetwork()
    x = torch.randn(1, 3, 224, 224)  # Input image
    out = model(x)
    print(out.shape)