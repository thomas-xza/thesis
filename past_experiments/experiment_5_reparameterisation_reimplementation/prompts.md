What Bayesian neural network things can you do with PyTorch without 3rd party libraries?


Gemini said
I can run a search to see what is currently possible with PyTorch regarding Bayesian Neural Networks without needing third-party libraries.
While PyTorch does not have dedicated, out-of-the-box Bayesian layers (like nn.BayesianLinear) in its native API, it provides all the mathematical primitives and automatic differentiation tools you need to build Bayesian Neural Networks (BNNs) from scratch.

If you want to strictly avoid third-party libraries like Pyro, torchbnn, or bayesian-torch, here are the main ways you can implement Bayesian concepts using pure vanilla PyTorch.

1. Monte Carlo (MC) Dropout: The Easiest BNN Approximation
The simplest way to get Bayesian-like uncertainty estimates without writing complex custom layers is through MC Dropout. As shown in the seminal paper by Yarin Gal and Zoubin Ghahramani, applying dropout before every weight layer is mathematically equivalent to an approximation of a probabilistic deep Gaussian process.

You build a standard neural network with nn.Dropout, but instead of turning dropout off during evaluation, you leave it on.

How to do it in PyTorch:

Create a standard model with dropout layers.

During inference, do not use model.eval(), or explicitly force the dropout layers to stay in training mode.

Run the same input through the network multiple times (e.g., 50 times).

Calculate the mean (your prediction) and the variance (your uncertainty) of those 50 outputs.

Python
import torch
import torch.nn as nn

class SimpleDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Inference phase
model = SimpleDropoutNet()
model.train() # KEEP dropout active

x_test = torch.randn(1, 10)
predictions = [model(x_test).item() for _ in range(50)]

mean_prediction = sum(predictions) / len(predictions)
print(f"Prediction: {mean_prediction}, Uncertainty (spread): {max(predictions) - min(predictions)}")
2. Custom Bayesian Layers (The Reparameterization Trick)
To build a "true" BNN where every weight is a probability distribution rather than a single number, you must write custom nn.Module classes. You will need to use PyTorch's native torch.distributions module.

Because you cannot backpropagate through a random sample directly, you have to use the Reparameterization Trick. Instead of sampling weights w directly from a normal distribution N(μ,σ 
2
 ), you define:

w=μ+σ⊙ϵ
where ϵ∼N(0,1). This allows PyTorch's Autograd to track gradients back to μ and σ.

How to do it in PyTorch:
You must define mu and rho (an unconstrained parameter to ensure σ stays positive via a softplus function) as nn.Parameter.

Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Mean of the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        # Parameter to calculate standard deviation
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        # Same for biases
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    def forward(self, x):
        # Calculate standard deviation: sigma = log(1 + exp(rho))
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # Sample epsilon
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)
        
        # Reparameterization trick
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        
        return F.linear(x, weight, bias)
Note: To train this custom layer, you will also need to manually code a custom Loss Function that combines standard negative log-likelihood (NLL) with the Kullback-Leibler (KL) divergence between your weight distributions and your chosen prior.

3. Deep Ensembles
While strictly a frequentist approach, Deep Ensembles are widely considered the gold standard for uncertainty quantification in deep learning, often outperforming complex BNNs, and they require zero custom layers.

How to do it in PyTorch:

Initialize 5 to 10 identical vanilla neural networks (e.g., standard ResNets or MLPs).

Because PyTorch randomly initializes the weights differently for each model, they will converge to different local minima.

Train each model independently on the same dataset.

At inference, pass the data through all 5 models. The variance between the 5 predictions represents your epistemic uncertainty (what the model doesn't know).

4. Stochastic Gradient Langevin Dynamics (SGLD)
You can turn standard training into a Bayesian posterior sampling process by modifying the optimizer. SGLD adds carefully scaled Gaussian noise to the gradients during the update step.

How to do it in PyTorch:
You can write a custom training loop or a custom torch.optim.Optimizer class. In a basic training loop using standard SGD, you would manually inject noise into the gradients before taking a step.

Python
# Pseudo-code for a manual SGLD step inside a training loop
learning_rate = 0.01

for param in model.parameters():
    if param.grad is not None:
        # Add gradient noise: N(0, 2 * learning_rate)
        noise = torch.randn_like(param.grad) * torch.sqrt(torch.tensor(2.0 * learning_rate))
        param.grad.add_(noise)

optimizer.step()
By doing this and saving the model weights at different epochs after burn-in, you collect samples from the posterior distribution.

Would you like a deeper dive into how to write the custom KL-Divergence loss function required to train the BayesianLinear layer in the second example?

__________

Rho is not being optimised during training, in Bayes-by-backprop implementation. Why?

Gemini said
In a Bayes-by-Backprop (BBB) implementation, if your ρ (rho) parameter—which controls the standard deviation (σ) of your weights—is not updating, it almost always comes down to either a broken computational graph or vanishing gradients.

Here are the four most common culprits in PyTorch and how to fix them.

1. You used .sample() instead of .rsample()
If you are using PyTorch's built-in torch.distributions to handle your weight priors and posteriors, this is the most frequent trap.

When you call .sample() on a distribution, PyTorch treats the output as a constant, effectively cutting off the computational graph. No gradients will flow back to μ or ρ.

The Fix: You must use the reparameterized sample method, .rsample(). This explicitly implements the reparameterization trick under the hood (w=μ+σ⋅ϵ), allowing Autograd to calculate the derivative with respect to σ (and therefore ρ).

2. Vanishing Gradients from the Softplus Initialization
In BBB, we usually define the standard deviation as a softplus function of ρ to ensure σ is always strictly positive:

σ=log(1+exp(ρ))
The derivative of the softplus function with respect to ρ is the logistic sigmoid function.
If you initialize ρ with a very large negative number (e.g., -10) to start with a small variance, the gradient becomes microscopically small (≈0.000045).

If ρ is initialized too low, the gradients vanish entirely, and the optimizer will never move the parameter, making it look like it's frozen.

The Fix: Check your initialization. Try initializing ρ closer to zero, for example, uniformly between -3 and -2.

3. Missing or Imbalanced KL-Divergence in the Loss (The ELBO)
Bayes-by-Backprop requires optimizing the Evidence Lower Bound (ELBO). Your loss function must be the sum of the Data Log-Likelihood (e.g., CrossEntropy, MSE) and the KL-Divergence between your weight posterior and your prior.

L=DataLoss+KL(q(w∣θ)∣∣p(w))
If you forgot the KL term, or if it is scaled incorrectly (e.g., not divided by the number of batches), the network's sole objective is to minimize the data error. To do this, the network wants to become entirely deterministic. It will actively push σ→0, driving ρ toward −∞. Once ρ gets negative enough, the softplus gradient dies (as explained in point #2), and ρ permanently stalls.

The Fix: Ensure your custom loss function correctly calculates and adds the KL divergence penalty for every single forward pass.

4. Parameter Registration Issue
PyTorch's optimizer only updates tensors that are explicitly registered as learnable parameters. If you created ρ as a standard tensor, optimizer.step() will ignore it.

The Fix: Ensure you are using nn.Parameter.

Python
# BAD: Optimizer won't see this
self.weight_rho = torch.tensor([-3.0]) 

# GOOD: Optimizer will update this
self.weight_rho = nn.Parameter(torch.Tensor([-3.0]))
How to debug this right now
You can definitively find out where the chain is breaking by checking the gradients immediately after calling loss.backward() but before optimizer.step(). Add this print statement to your training loop:

Python
loss.backward()

# Check if rho actually has a gradient tensor
print("Rho grad exists:", model.layer_name.weight_rho.grad is not None)

# Check if the gradient is entirely zeros (vanishing)
if model.layer_name.weight_rho.grad is not None:
    print("Rho grad sum:", model.layer_name.weight_rho.grad.abs().sum().item())

optimizer.step()
Would you like me to review the snippet of code where you define your forward pass or your custom ELBO loss function to spot the exact disconnect?

__________


Implement Bayes-by-backprop using .rsample()

Gemini said
To implement Bayes-by-Backprop using .rsample(), we use torch.distributions. This is the cleanest way to handle the reparameterization trick while ensuring that the gradients for both μ (mean) and ρ (which defines variance) are tracked correctly.

The Implementation
This example creates a Bayesian Linear layer where the weights and biases are treated as Normal distributions.

Python
import torch
import torch.nn as nn
import torch.distributions as dist

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters for weights
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))
        
        # Variational parameters for biases
        self.b_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.b_rho = nn.Parameter(torch.Tensor(out_features).fill_(-3.0))

        # Fixed Prior: Normal(0, prior_sigma)
        self.prior = dist.Normal(0, prior_sigma)
        
        # Placeholders for the KL divergence of this layer
        self.kl_loss = 0

    def forward(self, x):
        # 1. Transform rho to sigma (ensure positivity)
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))

        # 2. Define the Posterior distributions
        q_w = dist.Normal(self.w_mu, w_sigma)
        q_b = dist.Normal(self.b_mu, b_sigma)

        # 3. Reparameterized sampling
        # .rsample() uses (mu + sigma * epsilon) internally
        w = q_w.rsample()
        b = q_b.rsample()

        # 4. Calculate KL Divergence: KL(q||p)
        # Summing over all dimensions to get a scalar for this layer
        kl_w = dist.kl_divergence(q_w, self.prior).sum()
        kl_b = dist.kl_divergence(b_prior, self.prior).sum() # Assuming same prior for bias
        self.kl_loss = kl_w + kl_b

        return torch.nn.functional.linear(x, w, b)

# Helper to aggregate KL from all Bayesian layers
def get_total_kl(model):
    return sum(layer.kl_loss for layer in model.modules() if hasattr(layer, 'kl_loss'))
Key Technical Details
dist.kl_divergence(q, p): This is much more numerically stable than calculating log(q)−log(p) manually. It uses the closed-form analytical solution for two Gaussians.

Initialization of rho: Setting rho to -3.0 results in an initial σ≈0.05. If you initialize it too low (e.g., -10), the gradient of the softplus becomes so small that ρ will never update (the vanishing gradient problem you likely encountered).

.rsample() vs .sample(): Using .sample() would break the connection to w_mu and w_rho. .rsample() ensures that when you call loss.backward(), the chain rule reaches your parameters.

Training with the ELBO Loss
To train this, your loss function must combine the prediction error (NLL) and the KL Divergence.

Python
# Training Loop Logic
model = nn.Sequential(BayesianLinear(10, 5), nn.ReLU(), BayesianLinear(5, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss(reduction='sum') # Sum is standard for ELBO

for x, y in dataloader:
    optimizer.zero_grad()
    
    output = model(x)
    
    # 1. Likelihood loss (Data term)
    data_loss = criterion(output, y)
    
    # 2. Complexity loss (KL term)
    # Note: In practice, KL is often scaled by (1 / total_batches)
    kl_loss = get_total_kl(model)
    
    # 3. Total Loss (ELBO)
    loss = data_loss + kl_loss
    
    loss.backward()
    optimizer.step()
Why this fixes the ρ issue
By using dist.Normal(...).rsample(), PyTorch builds a graph where the output w is a direct function of w_rho. Because the kl_divergence also depends on w_rho, the parameter receives gradients from two sources:

The Prior push: Trying to make the distribution look like the prior (usually pulling σ larger).

The Likelihood pull: Trying to make the weights deterministic to fit the data (pulling σ smaller).

Would you like to see how to implement a Scale Mixture Prior (Spike-and-Slab) to further improve the Bayesian sparsity of the model?

__________


Implement Bayes-by-backprop using .rsample()

Gemini said
To implement Bayes-by-Backprop effectively using .rsample(), we utilize torch.distributions.Normal. This approach is cleaner than manual reparameterization because it handles the math under the hood while ensuring the gradient chain remains intact for both μ and ρ.

The Bayesian Linear Layer
In this implementation, every forward pass generates a new stochastic weight sample. The KL divergence is calculated analytically during the forward pass and stored so the optimizer can access it.

Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. Variational parameters (Mean and Rho)
        # We initialize rho around -3 to -5 so initial sigma is small
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))
        
        self.b_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.b_rho = nn.Parameter(torch.Tensor(out_features).fill_(-3.0))

        # 2. Prior Distribution (Fixed)
        self.prior_sigma = prior_sigma
        # We define a Scale-Location distribution for the prior
        self.prior = dist.Normal(0, prior_sigma)
        
        self.kl_loss = 0

    def forward(self, x):
        # 3. Transform rho to sigma: sigma = log(1 + exp(rho))
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        # 4. Define Posterior Distributions
        q_w = dist.Normal(self.w_mu, w_sigma)
        q_b = dist.Normal(self.b_mu, b_sigma)

        # 5. Reparameterized Sampling (The Magic)
        # .rsample() ensures gradients flow to mu and rho
        w = q_w.rsample()
        b = q_b.rsample()

        # 6. Calculate KL Divergence: KL(q||p)
        # This is the "Complexity Cost"
        kl_w = dist.kl_divergence(q_w, self.prior).sum()
        kl_b = dist.kl_divergence(q_b, self.prior).sum()
        self.kl_loss = kl_w + kl_b

        return F.linear(x, w, b)
Training with the ELBO Loss
When training a BNN, your total loss is the Evidence Lower Bound (ELBO). It consists of the Likelihood (how well you fit the data) and the KL Divergence (how well you stay near the prior).

Crucial Scaling Note: Because the Likelihood is usually calculated per batch, but the KL divergence is a property of the whole model, you must scale the KL term by  
M
1
​
  (where M is the number of batches) to maintain the correct balance.

Python
def train_step(model, optimizer, data, target, num_batches):
    optimizer.zero_grad()
    
    # Forward pass (stochastic)
    output = model(data)
    
    # 1. Negative Log Likelihood (NLL) / MSE
    # Use reduction='sum' to match the ELBO derivation
    data_loss = F.mse_loss(output, target, reduction='sum')
    
    # 2. Total KL from all Bayesian layers
    total_kl = sum(layer.kl_loss for layer in model.modules() if hasattr(layer, 'kl_loss'))
    
    # 3. ELBO Loss (scaled for the current batch)
    loss = data_loss + (total_kl / num_batches)
    
    loss.backward()
    optimizer.step()
    return loss.item()
Why this fixes the "Rho Not Optimizing" issue
Direct Gradient Path: By using dist.Normal(...).rsample(), PyTorch tracks the standard deviation through the softplus back to rho.

Explicit KL Gradient: Since kl_divergence is calculated using q_w (which contains rho), the backward pass specifically tells rho how to move to satisfy the prior.

Numerical Stability: F.softplus is much safer than writing torch.log(1 + torch.exp(rho)) manually, as it handles large values of rho without overflowing.

Pro-Tip: Multi-Sample Inference
At test time, the model is still stochastic. To get a robust prediction, you should run the input through the model multiple times and average the results:

Python
model.eval()
with torch.no_grad():
    # Take 50 samples of the posterior
    samples = [model(test_input) for _ in range(50)]
    samples = torch.stack(samples)
    
    mean_prediction = samples.mean(dim=0)
    uncertainty = samples.std(dim=0)
Would you like to see how to implement a Scale Mixture Prior (Spike-and-Slab) to help the model zero-out redundant weights?