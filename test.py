import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gspn import GSPNLayer

def test_gspn():
    # Create layer instance
    dim = 64
    gspn = GSPNLayer(dim=dim, is_global=True)
    
    # Create sample input
    batch_size = 2
    height = 32
    width = 32
    x = torch.randn(batch_size, dim, height, width)
    
    # Forward pass
    output = gspn(x)
    
    # Validate output shape
    assert output.shape == (batch_size, dim, height, width)
    
    # Test local version
    gspn_local = GSPNLayer(dim=dim, is_global=False, group_size=2)
    output_local = gspn_local(x)
    assert output_local.shape == (batch_size, dim, height, width)

    print("Shapes:")
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Local Output: {output_local.shape}")
    
    # Test parameter counts
    total_params = sum(p.numel() for p in gspn.parameters())
    print(f"\nTotal parameters: {total_params}")

if __name__ == "__main__":
    test_gspn()