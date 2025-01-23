import torch
import torch.nn as nn
import torch.nn.functional as F

from .grn import GRN

# My shot at implementing "Parallel Sequence Modeling via Generalized Spatial Propagation Network" https://arxiv.org/pdf/2501.12381

# Some random notes, adding GRN's into this layer structure did not seem as successful as adding it to the upper blocks in example network. In fact, it performed worse to add them here.

class GSPNLayer(nn.Module):
    def __init__(self, dim, is_global=True, group_size=2):
        super().__init__()
        # Dimension reduction and projection layers following Section 4.2
        self.dim_reduce = nn.Conv2d(dim, dim // 4, 1)  # Reduces channel dimension for efficiency
        
        self.to_u = nn.Conv2d(dim // 4, dim, 1)        # Projects to u in Eq 2
        self.to_lambda = nn.Conv2d(dim // 4, dim, 1)   # Projects to λ in Eq 1
        self.to_weights = nn.Conv2d(dim // 4, 3, 1)    # Generates tri-diagonal weights w in Eq 1
        
        # Merging layers for 4-directional integration | Section 3.3
        self.learned_merge_weights = nn.Conv2d(dim * 4, 4, 1)
        self.merge = nn.Conv2d(dim * 4, dim, 1)
        self.is_global = is_global
        self.group_size = group_size
        
    def build_tridiagonal(self, pre_weights, prop_seq_len, propagation_dim):
        b, _, h, w = pre_weights.shape
        
        matrix = torch.zeros(b, h, prop_seq_len, prop_seq_len, device=pre_weights.device)
        
        # Normalize weights row-wise using sigmoid and row-wise normalization as in Eq 6
        weights = torch.sigmoid(pre_weights)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        # Fill the tridiagonal matrix using a sliding window approach
        # The fundamental operation feels like a parallelizable convolution.
        for i in range(prop_seq_len):
            for offset, k in zip([-1, 0, 1], range(3)):
                j = i + offset
                if 0 <= j < prop_seq_len:
                    if propagation_dim == 2:  # Propagate along height
                        matrix[:, :, i, j] = weights[:, k, :, i]
                    elif propagation_dim == 3:  # Propagate along width
                        matrix[:, :, i, j] = weights[:, k, i, :]
        
        return matrix.unsqueeze(1)

    def propagate_direction(self, x, direction):
        b, c, h, w = x.shape
        column_transposed = direction in ['lr', 'rl']
        reverse = direction in ['bt', 'rl']
        
        # Determine propagation dimension and sequence length
        if direction in ['tb', 'bt']:
            prop_seq_len = h
            propagation_dim = 2  # Propagate along height
        else:
            prop_seq_len = w
            propagation_dim = 3  # Propagate along width

        # Dimension reduction and projection as described in Section 4.2
        reduced = self.dim_reduce(x)        
        u = self.to_u(reduced)  # Output weights u from Eq. 2
        lambda_weights = self.to_lambda(reduced)  # λ weights from Eq. 1
        pre_weights = self.to_weights(reduced)  # Pre-weights for tridiagonal matrix

        # Transpose for column-wise directions
        if column_transposed:
            x = x.transpose(-1, -2)
            lambda_weights = lambda_weights.transpose(-1, -2)
            pre_weights = pre_weights.transpose(-1, -2)
            u = u.transpose(-1, -2)

        # Reverse sequence for backward directions
        if reverse:
            x = torch.flip(x, dims=[propagation_dim])
            lambda_weights = torch.flip(lambda_weights, dims=[propagation_dim])
            pre_weights = torch.flip(pre_weights, dims=[propagation_dim])
            u = torch.flip(u, dims=[propagation_dim])

        weights = self.build_tridiagonal(pre_weights, prop_seq_len, propagation_dim)

        hidden = torch.zeros(b, c, *x.shape[3:], device=x.device)
        propagation = []
        for i in range(prop_seq_len):
            curr_w = weights[:, :, :, i, :]
            curr_lambda = lambda_weights[:, :, :, i] if direction in ['lr', 'rl'] else lambda_weights[:, :, i]
            curr_x = x[:, :, i] if direction in ['lr', 'rl'] else x[:, :, :, i]
            
            # Apply propagation equation hi = wi·hi-1 + λi⊙xi from Eq. 1
            hidden = (hidden.unsqueeze(-1) * curr_w).sum(dim=-1) + curr_lambda * curr_x
            propagation.append(hidden)

        # Eq 2
        output = torch.stack(propagation, dim=2 if direction in ['lr', 'rl'] else 3)
        
        # Unreversed any reverses
        if reverse:
            output = torch.flip(output, dims=[propagation_dim])

        # Apply output weights u and transpose back for column-wise directions
        output = output * u
        if column_transposed:
            output = output.transpose(-1, -2)
                
        return output

    def forward(self, x):
        # input B C H W
        # 4-Directional propagation | Section 3.3
        outputs = [self.propagate_direction(x, d) for d in ['tb', 'bt', 'lr', 'rl']]
        concat = torch.cat(outputs, dim=1)
        
        # Section 4.2
        weights = self.learned_merge_weights(concat)
        weights = F.softmax(weights, dim=1).unsqueeze(1)
        
        # Merge outputs from all directions
        stacked = torch.stack(outputs, dim=2)
        merged = (stacked * weights).sum(dim=2)
        
        output = self.merge(concat)

        return output