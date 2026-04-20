#!/usr/bin/env/python3

##  Layer extracted from bayesian-torch library.

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

torch.set_default_dtype(torch.bfloat16)

class Linear_gaussian_reparam(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.25,
                 prior_variance=0,
                 posterior_mu_init=0.25,
                 posterior_rho_init=-10,
                 bias=True):
        
        """
        Implements Linear layer with reparameterisation.

        Note that posterior_rho_init will generate sigma via:
        ln(1 + exp(rho)) = sigma

        And therefore:
        rho = ln(exp(sigma)) - 1)

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        
        super().__init__()

        ##  Standard linear layer features.

        self.in_features = in_features
        
        self.out_features = out_features

        ##  The prior distributions of the weights.
        
        self.prior_mean = prior_mean
        
        self.prior_variance = prior_variance

        ##  The initial distribution parameters of the distributions
        ##  to be trained.
        
        ##  This would typically be equivalent to the prior, but a
        ##  separate variables allows a pretrained drop-in.
        
        ##  sigma = log (1 + exp(rho))
        
        self.posterior_mu_init = posterior_mu_init
        
        self.posterior_rho_init = posterior_rho_init
        
        self.bias = bias

        ##  Setup training distribution parameters.

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)

        ##  Register buffer: "data that is not a model parameter but
        ##  that is part of the module's state". E.g. during batch
        ##  normalisation, or calculating KL loss.

        ##  eps_weight, epsilon weight, is used to generate weights
        ##  during forward propagation.
        
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        ##  Prior weight mu/sigma are used to calculate KL divergence.
        
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)

        ##  Optionally also generate bias parameter probability
        ##  distributions.

        if bias:
            
            self.mu_bias = Parameter(torch.Tensor(out_features), requires_grad=True)
            
            self.rho_bias = Parameter(torch.Tensor(out_features), requires_grad=True)
            
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
            
        else:
            
            self.register_buffer('prior_bias_mu', None, persistent=False)
            
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            
            self.register_parameter('mu_bias', None)
            
            self.register_parameter('rho_bias', None)
            
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()
        
        ##  self.quant_prepare=False

        
    def init_parameters(self):

        ##  Fill tensor with value.
        
        self.prior_weight_mu.fill_(self.prior_mean)
        
        self.prior_weight_sigma.fill_(self.prior_variance)

        ##  Fill tensor with Gaussian distribution samples.
        
        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0)
        
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)

        # print('----------------------------  MU')
        # print(self.mu_weight)

        # print('---------------------------- RHO')
        # print(self.rho_weight)

        ##  Repeat for bias values.
        
        if self.mu_bias is not None:
            
            self.prior_bias_mu.fill_(self.prior_mean)
            
            self.prior_bias_sigma.fill_(self.prior_variance)
            
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            
            self.rho_bias.data.normal_(mean=self.posterior_rho_init,
                                       std=0.1)

        
    def prepare(self):

        ##  Quantization regards operating at a lower precision,
        ##  e.g. FP8.
        
        pass

    #     self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
    #                                      QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
    
    #     self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
    #                                      QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        
    #     self.dequant = torch.quantization.DeQuantStub()
    
    #     self.quant_prepare=True

    
    def kl_loss(self):

        ##  Generate sigma for weight generation via trained parameter.
        
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        ##  Find KL divergence between prior distribution and current
        ##  distribution.
        
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)

        ##  Potentially also involve bias parameters in KL divergence
        ##  loss score.
        
        if self.mu_bias is not None:
            
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
            
        return kl

    
    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        
        return kl.mean()


    def forward(self, input, return_kl=True):

        ##  Generate sigma for weight generation via trained parameter.
        
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
 
        # print('------------------------------------------------------------  RHO WEIGHT')
        
        # print(self.rho_weight)

        ##  Generate epsilon weight via Gaussian distribution, N(0, 1).
        
        eps_weight = self.eps_weight.data.normal_(mean=0, std=0.000)

        ##  Generate weight via sigma and epsilon weights.
        
        tmp_result = sigma_weight * eps_weight
        
        weight = self.mu_weight #+ tmp_result

        # print('------------------------------------------------------------  WEIGHTS GEN')
        
        # print(weight)

        if return_kl:

            ##  Find difference between newly generated approximation
            ##  weight function and prior weight function.
            
            kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)
            
        bias = None

        if self.mu_bias is not None:

            ##  Generate bias parameter via epsilon.
            
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())

            ##  Generate KL loss value related to current bias
            ##  approximation distribution, prior distribution.

            if return_kl:
                kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        ##  Run input through standard linear layer.

        out = F.linear(input, weight, bias)

        # if self.quant_prepare:
        #     # quint8 quantstub
        #     input = self.quint_quant[0](input) # input
        #     out = self.quint_quant[1](out) # output

        #     # qint8 quantstub
        #     sigma_weight = self.qint_quant[0](sigma_weight) # weight
        #     mu_weight = self.qint_quant[1](self.mu_weight) # weight
        #     eps_weight = self.qint_quant[2](eps_weight) # random variable
        #     tmp_result =self.qint_quant[3](tmp_result) # multiply activation
        #     weight = self.qint_quant[4](weight) # add activatation

        if self.mu_bias is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        # print(kl)

            # return out, kl

        return out
