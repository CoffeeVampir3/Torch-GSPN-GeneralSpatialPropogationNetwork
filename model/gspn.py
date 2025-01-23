import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# My shot at implementing "Parallel Sequence Modeling via Generalized Spatial Propagation Network" https://arxiv.org/pdf/2501.12381
# There's some omitted things, for example I just used basic conv's there's no GRNs as indicated in the paper

class GSPNLayer(nn.Module):
    def __init__(self, dim, is_global=True, group_size=2):
        super().__init__()
        # Dimension reduction and projection layers following Section 4.2
        self.dim_reduce = nn.Conv2d(dim, dim // 4, 1)  # Reduces channel dimension for efficiency
        self.to_u = nn.Conv2d(dim // 4, dim, 1)        # Projects to u in Eq. (2)
        self.to_lambda = nn.Conv2d(dim // 4, dim, 1)   # Projects to λ in Eq. (1)
        self.to_weights = nn.Conv2d(dim // 4, 3, 1)    # Generates tri-diagonal weights w in Eq. (1)
        
        # Merging layers for 4-directional integration as discussed in Section 3.3
        self.merge_weights = nn.Conv2d(dim * 4, 4, 1)
        self.merge = nn.Conv2d(dim * 4, dim, 1)
        self.is_global = is_global
        self.group_size = group_size

    def build_tridiagonal(self, pre_weights, prop_seq_len):
        b, _, h, _ = pre_weights.shape  # [b, 3, h, w]
        
        matrix = torch.zeros(b, h, prop_seq_len, prop_seq_len, device=pre_weights.device)  # [b, h, prop_seq_len, prop_seq_len]
        
        # Eq. 6 row stochastic normalization
        weights = torch.sigmoid(pre_weights)  # [b, 3, h, w]
        # Normalize weights row-wise so each row sums to 1
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [b, 3, h, w]
        
        # Fill the tridiagonal matrix with normalized weights
        for i in range(prop_seq_len):
            for offset, k in zip([-1, 0, 1], range(3)):  # 3-way connection: left, middle, right
                j = i + offset  # Tri-Neighbor index
                if 0 <= j < prop_seq_len:
                    matrix[:, :, i, j] = weights[:, k, :, i]
        
        return matrix.unsqueeze(1)  # [b, 1, h, prop_seq_len, prop_seq_len]

    def propagate_direction(self, x, direction):
        b, c, h, w = x.shape
        
        #print(F"X: {x.shape}")
        reduced = self.dim_reduce(x)  # Downsample dimensions to operate on smaller target
        u = self.to_u(reduced)  # Output weights u from Eq. (2)
        lambda_weights = self.to_lambda(reduced)  # λ weights from Eq. (1)
        pre_weights = self.to_weights(reduced)

        if direction in ['tb', 'bt']:
            hidden = torch.zeros(b, c, w, device=x.device)  # Hidden states for row-wise propagation
            seq_len = h  # Propagate along height dimension
        else:
            hidden = torch.zeros(b, c, h, device=x.device)  # Hidden states for column-wise propagation
            seq_len = w  # Propagate along width dimension
            x = x.transpose(-1, -2)  # Transpose for column-wise propagation
            lambda_weights = lambda_weights.transpose(-1, -2)
            pre_weights = pre_weights.transpose(-1, -2)
            u = u.transpose(-1, -2)

        # Build tridiagonal weight matrix
        weights = self.build_tridiagonal(pre_weights, seq_len) # [b, 1, h, width, width]

        # Implements the "linear recurrent process" from Eq. (1)
        # I'm not entirely clear if I did this correctly but gave my best interpretation.
        propagation = []
        for i in range(seq_len):
            curr_w = weights[:, :, :, i, :] # Shape: [b, 1, h, w]
            #print(F"cw: {curr_w.shape}")
            curr_lambda = lambda_weights[:, :, i] # [b, c, seq_len]
            #print(F"cl: {curr_lambda.shape}")
            curr_x = x[:, :, i] # [b, c, seq_len]
            #print(F"cx: {curr_x.shape}")

            # Apply propagation equation hi = wi·hi-1 + λi⊙xi from Eq. (1)
            hidden = (hidden.unsqueeze(-1) * curr_w).sum(dim=-1) + curr_lambda * curr_x
            propagation.append(hidden)

        # Stack propagated hidden states and apply output weights (Eq. 2)
        output = torch.stack(propagation, dim=2)
        return output * u  # Apply output weights u

    def forward(self, x):
        # 4-D Scan Propagation from all directions
        outputs = [self.propagate_direction(x, d) for d in ['tb', 'bt', 'lr', 'rl']]
        concat = torch.cat(outputs, dim=1)
        
        # Learned merging weights | Section 4.2
        weights = self.merge_weights(concat)
        weights = F.softmax(weights, dim=1).unsqueeze(1)
        
        # Merge outputs from all directions
        stacked = torch.stack(outputs, dim=2)
        merged = (stacked * weights).sum(dim=2)
        
        return self.merge(concat)