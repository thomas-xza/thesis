# mypy: allow-untyped-defs

##  Layer extracted from PyTorch library.

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module


__all__ = [
    "Linear_deterministic",
]


class Linear_reparam_gaussian(Module):
       
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    weight_mu: Tensor
    weight_rho: Tensor
    bias_rho: Tensor

    def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device = None,
            dtype = None,
            w_mu_init: float = 0.25,
            w_rho_init: float = -5,
            b_mu_init: float = 0.25,
            b_rho_init: float = -2.5
    ) -> None:
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        super().__init__()
        
        self.in_features = in_features
        
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.weight_mu = Parameter(torch.empty(out_features,
                                               in_features, **factory_kwargs))
        
        self.weight_rho = Parameter(torch.empty(out_features,
                                                in_features, **factory_kwargs))

        # self.weight_sigma = Parameter(torch.empty(out_features,
        #                                         in_features, **factory_kwargs))

        self.register_buffer('weight_eps', torch.empty(out_features,
                                                        in_features,
                                                        **factory_kwargs),
                                                        persistent=False)

        if bias:
                        
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.bias_mu = Parameter(torch.empty(out_features, **factory_kwargs))
        
            self.bias_rho = Parameter(torch.empty(out_features, **factory_kwargs))
            
            # self.bias_sigma = Parameter(torch.empty(out_features, **factory_kwargs))
            
            self.register_buffer('bias_eps',
                             torch.empty(out_features, in_features, **factory_kwargs),
                                 persistent=False)

        else:
            
            self.register_parameter("bias", None, **factory_kwargs)
            
            self.register_parameter("bias_mu", None, **factory_kwargs)
            
            self.register_parameter("bias_rho", None, **factory_kwargs)
            
            # self.register_parameter("bias_sigma", None, **factory_kwargs)
            
            self.register_buffer('bias_eps', None, persistent=False)

        self.reset_parameters(w_mu_init, w_rho_init, b_mu_init, b_rho_init)

        
    def reset_parameters(self, w_mu_init: float, w_rho_init: float, b_mu_init: float, b_rho_init: float) -> None:

        # init.constant_(self.weight_mu, mu_init)
 
        # init.constant_(self.weight_rho, rho_init)
 
        init.uniform_(self.weight_mu, a = - w_mu_init, b = w_mu_init)
        
        init.uniform_(self.weight_rho, a = w_rho_init, b = w_rho_init)
        
        # init.uniform_(self.weight_sigma, a = 0.00672, b = 0.00672)
        
        init.uniform_(self.bias_mu, a = b_rho_init, b = b_rho_init)
        
        init.uniform_(self.bias_rho, a = b_rho_init, b = b_rho_init)

        # print(torch.log1p(torch.exp(self.weight_rho)))
        
        # init.uniform_(self.bias_sigma, a = rho_init, b = rho_init)
        
        # init.kaiming_uniform_(self.weight_rho, a=math.sqrt(5))

        ##  Experiment 2 initialised the weight tensor via Kaiming,
        ##  then the bias via the weight tensor.
        
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # if self.bias is not None:
            
        #     # init.constant_(self.bias_mu, mu_init)
        
        #     # init.constant_(self.bias_rho, rho_init)
        
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            
        #     init.uniform_(self.bias, -bound, bound)

            
    def forward(self, input: Tensor) -> Tensor:

        self.weight_eps = self.weight_eps.data.normal_(mean = 0, std = 1)

        self.bias_eps = self.bias_eps.data.normal_(mean = 0, std = 0.3)

        weight_sigma = F.softplus(self.weight_rho)
        
        bias_sigma = F.softplus(self.bias_rho)

        weight_epsilon = torch.randn_like(weight_sigma)
        
        bias_epsilon = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * weight_epsilon
        
        bias = self.bias_mu + bias_sigma * bias_epsilon

        # print(self.weight_sigma)

        # self.weight_eps = torch.abs(self.weight_eps)
        
        return F.linear(input, weight, bias)

    
    def extra_repr(self) -> str:
        
        """
        Return the extra representation of the module.
        """
        
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
