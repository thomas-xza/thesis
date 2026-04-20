#!/usr/bin/env/python3

import torch
from torch import nn


class Linear_model(nn.Module):
    
    def __init__(self, n: int):
        
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

        nn.init.uniform_(self.linear_relu_stack[0].weight, a=-0.25, b=0.25)
        nn.init.uniform_(self.linear_relu_stack[2].weight, a=-0.25, b=0.25)
        nn.init.uniform_(self.linear_relu_stack[4].weight, a=-0.25, b=0.25)
        

    def forward(self, x: torch.Tensor):
        
        logits = self.linear_relu_stack(x)

        # print(logits)

        out = torch.sigmoid(logits)
        
        return out
