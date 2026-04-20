
class Bayesian_linear(nn.Module):
    
    def __init__(self, in_features, out_features, prior_w_sigma = 1.0, prior_b_sigma = 1.0):
        
        super().__init__()
        
        self.in_features = in_features
        
        self.out_features = out_features

        # 1. Variational parameters (Mean and Rho)
        
        ##  We initialize rho around -3 to -5 so initial sigma is small
        
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))
        
        self.b_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        
        self.b_rho = nn.Parameter(torch.Tensor(out_features).fill_(-3.0))

        self.prior_w_sigma = prior_w_sigma
        
        self.prior_b_sigma = prior_b_sigma
        
        self.prior_w = dist.Normal(0, prior_w_sigma)
        
        self.prior_b = dist.Normal(0, prior_b_sigma)
        
        self.kl_loss = 0

        
    def forward(self, x):

        ##  PyTorch built-ins used as much as possible in forward().
        
        ##  sigma = log(1 + exp(rho))
        
        w_sigma = F.softplus(self.w_rho)
        
        b_sigma = F.softplus(self.b_rho)

        ##  Sample approximation function based on Parameter objects.
        
        q_w = dist.Normal(self.w_mu, w_sigma)
        
        q_b = dist.Normal(self.b_mu, b_sigma)

        ##  PyTorch official reparameterisation implementation.
        
        w = q_w.rsample()
        
        b = q_b.rsample()

        ##  Recalculate KL Divergence: KL(q||p).
        
        kl_w = dist.kl_divergence(q_w, self.prior).sum()
        
        kl_b = dist.kl_divergence(q_b, self.prior).sum()
        
        self.kl_loss = kl_w + kl_b

        return F.linear(x, w, b)
