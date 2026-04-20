#!/usr/bin/env python3

import torch
import torch.nn as nn

class Parameterised_sigmoid(nn.Module):

    def __init__(self, k_init = 75):
        
        super().__init__()
        
        self.k = nn.Parameter(torch.tensor([k_init]))

        
    def forward(self, x):

        return torch.minimum(x, torch.zeros_like(x)) - torch.log1p(torch.exp(-torch.abs(x)))

        # return 1 / (1 + torch.exp(-self.k * x))

        # const auto min = std::min(0.0, in);
        # We use torch.minimum to compare the tensor against the scalar 0
        val_min = torch.minimum(torch.zeros_like(input_tensor), input_tensor)
    
        # const auto z = std::exp(-std::abs(in));
        z = torch.exp(-torch.abs(input_tensor))
    
        # return min - std::log1p(z);
        return val_min - torch.log1p(z)
        
