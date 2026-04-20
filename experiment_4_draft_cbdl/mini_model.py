#!/usr/bin/env/python3
import torch
from torch import nn


class Linear_model(nn.Module):
    
    def __init__(self, n: int, udist: [int, int], random_seed: int):
        
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

        if random_seed != 239852:

            for layer in self.linear_relu_stack:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)

        else:
        
            nn.init.uniform_(self.linear_relu_stack[0].weight, a=-udist[0], b=udist[1])
            nn.init.uniform_(self.linear_relu_stack[2].weight, a=-udist[0], b=udist[1])
            nn.init.uniform_(self.linear_relu_stack[4].weight, a=-udist[0], b=udist[1])
        

    def forward(self, x: torch.Tensor):
        
        logits = self.linear_relu_stack(x)

        # print(logits)

        out = torch.sigmoid(logits)
        
        return out
