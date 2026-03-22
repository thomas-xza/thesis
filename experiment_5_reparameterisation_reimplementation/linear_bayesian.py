#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class Linear_bayesian(nn.Module):

    
    def __init__(self, in_features, out_features, w_mu_init = -0.05, b_mu_init = -0.05, w_rho_init = -3.0, b_rho_init = -3.0):
        
        super().__init__()
        
        self.in_features = in_features
        
        self.out_features = out_features

        ##  Generate dynamic tensors for approximation posterior.
        
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).fill_(w_mu_init))
        
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(w_rho_init))
        
        self.b_mu = nn.Parameter(torch.Tensor(out_features).fill_(b_mu_init))
        
        self.b_rho = nn.Parameter(torch.Tensor(out_features).fill_(b_rho_init))

        ##  Generate static tensors for prior distribution.

        ##  F.softplus(rho) = log(1 + exp(rho))

        self.register_buffer(
            "prior_w_sigma",
            F.softplus(torch.Tensor(out_features, in_features).fill_(w_rho_init))
        )

        self.register_buffer(
            "prior_b_sigma",
            F.softplus(torch.Tensor(out_features, in_features).fill_(b_rho_init))
        )

        self.register_buffer(
            "prior_w_mu",
            torch.Tensor(out_features, in_features).fill_(0)
        )

        self.register_buffer(
            "prior_b_mu",
            torch.Tensor(out_features, in_features).fill_(0)
        )

        self.kl_loss = 0


    def kl(self):

        return self.kl_loss

        
    def forward(self, x):

        ##  Setup distribution based on sigma values.
        ##  Despite being static in value, will only be on correct
        ##  device if defined within forward() pass.
        
        self.prior_w_dist = dist.Normal(self.prior_w_mu, self.prior_w_sigma)

        self.prior_b_dist = dist.Normal(self.prior_b_mu, self.prior_b_sigma)
        
        ##  PyTorch built-ins used as much as possible, in this
        ##  implementation.
        
        ##  sigma = log(1 + exp(rho))

        ##  Generate sigma based on current values of rho.
        
        w_sigma = F.softplus(self.w_rho)
        
        b_sigma = F.softplus(self.b_rho)

        print(w_sigma.sum())

        ##  Setup approximation function based on Parameter objects.
        
        q_w = dist.Normal(self.w_mu, w_sigma)
        
        q_b = dist.Normal(self.b_mu, b_sigma)

        ##  Sample via PyTorch official reparameterisation
        ##  implementation.
        
        w = q_w.rsample()
        
        b = q_b.rsample()

        ##  Recalculate KL divergence: KL(q || p).

        kl_w = dist.kl_divergence(q_w, self.prior_w_dist).sum()
        
        kl_b = dist.kl_divergence(q_b, self.prior_b_dist).sum()

        ##  Note that this final result will be based on the full
        ##  batch size.
        
        # self.kl_loss = kl_w + kl_b

        return F.linear(x, w, b), (kl_w + kl_b)

