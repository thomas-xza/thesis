#!/usr/bin/env/python3

import torch
from torch import nn

from linear_layer_deterministic import Linear_deterministic


class Linear_model(nn.Module):
    
    def __init__(self, n: int):
        
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            Linear_deterministic(1, n),
            nn.ReLU(),
            Linear_deterministic(n, n),
            nn.ReLU(),
            Linear_deterministic(n, 1)
        )

        nn.init.uniform_(self.linear_relu_stack[0].weight, a=-0.25, b=0.25)
        nn.init.uniform_(self.linear_relu_stack[2].weight, a=-0.25, b=0.25)
        nn.init.uniform_(self.linear_relu_stack[4].weight, a=-0.25, b=0.25)
        

    def forward(self, x: torch.Tensor):
        
        logits = self.linear_relu_stack(x)

        out = torch.sigmoid(logits)        
        # print(out, logits)

        return out
