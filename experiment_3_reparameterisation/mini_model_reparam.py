#!/usr/bin/env python3

import torch
from torch import nn

from linear_layer_reparam_v2 import Linear_reparam_gaussian
# from sigmoid_param import Parameterised_sigmoid

class Linear_model(nn.Module):
    
    def __init__(self, n: int, s_k: float, w_mu: float, w_rho: float):

        ##  n = neurons in 1-n-n-1 network
        
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            Linear_reparam_gaussian(1, n, w_mu_init = w_mu, w_rho_init = w_rho),
            nn.ReLU(),
            Linear_reparam_gaussian(n, n),
            nn.ReLU(),
            Linear_reparam_gaussian(n, 1),
            # Parameterised_sigmoid(s_k)
            nn.Sigmoid()
        )

        ##  Non-probabilistic equivalent:

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(1, n),
        #     nn.ReLU(),
        #     nn.Linear(n, n),
        #     nn.ReLU(),
        #     nn.Linear(n, 1)
        # )

        # nn.init.uniform_(self.linear_relu_stack[0].weight, a=-0.25, b=0.25)
        # nn.init.uniform_(self.linear_relu_stack[2].weight, a=-0.25, b=0.25)
        # nn.init.uniform_(self.linear_relu_stack[4].weight, a=-0.25, b=0.25)
        

    def forward(self, x: torch.Tensor):

        return self.linear_relu_stack(x)
    
