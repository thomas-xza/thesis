#!/usr/bin/env python3

import torch
from torch import nn

from reparam.linear_bayesian import Linear_bayesian
# from sigmoid_param import Parameterised_sigmoid

class Linear_model(nn.Module):
    
    def __init__(self, n: int, mu_w_init: float, mu_b_init: float, rho_w_init: float, rho_b_init: float):

        ##  n = neurons in 1-n-n-1 network
        
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            Linear_bayesian(1, n, mu_w_init, mu_b_init, rho_w_init, rho_b_init),
            nn.ReLU(),
            Linear_bayesian(n, n, mu_w_init, mu_b_init, rho_w_init, rho_b_init),
            nn.ReLU(),
            Linear_bayesian(n, 1, mu_w_init, mu_b_init, rho_w_init, rho_b_init),
            nn.Sigmoid()
        )

        self.params_n = self.calc_params_n(self.linear_relu_stack)

        self.kl_m = 0

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

        y_hat = self.linear_relu_stack(x)

        linear_layers = tuple(filter(
            lambda m: isinstance(m, Linear_bayesian), self.linear_relu_stack.modules()
        ))

        kl_layers = tuple(map(lambda l: l.kl(), linear_layers))

        kl_model = sum(kl_layers) / self.params_n

        self.kl_m = kl_model

        return y_hat, kl_model
    

    def calc_params_n(self, model: nn.Module):

        ##  Takes a model, and finds |θ|, that is, the quantity of weights
        ##  and biases across all layers.

        ##  Extract layers.

        linear_layers = tuple(filter(
            lambda x: isinstance(x, Linear_bayesian), model.modules()
        ))

        ##  Extract neurons in to each layer.

        features = tuple(map(lambda m: m.in_features, linear_layers))

        ##  Setup pairs of adjacent layers.

        pairs = list(map(lambda i: (features[i], features[i+1]), range(len(features)-1)))

        ##  Multiply each pair.

        products = list(map(lambda pair: pair[0] * pair[1], pairs))

        return sum(products)
