# Concept questions

I used AI to help with my research regarding uncertainties on the topics. 
**Question: How do you measure uncertainty in neural network?**

“uncertainty” in neural networks usually means _how sure is the model about its prediction_. There are two main types and several ways to estimate them.

---
## 1. Types of Uncertainty

1. **Aleatoric uncertainty**
    - Comes from noise in the data itself (irreducible).
    - Example: measuring position with a noisy sensor.
2. **Epistemic uncertainty**
    - Comes from lack of knowledge / limited data (reducible).
    - Example: model never saw this region in training → less confident.

---
## 2. Methods to Measure Uncertainty

### A. Bayesian Neural Networks (BNNs)
- Treat weights as **probability distributions**, not fixed values.
- Do inference with posterior distributions → output predictive distribution.
- In practice: use **variational inference** or **MCMC** to approximate.
- Gives principled epistemic + aleatoric uncertainty, but expensive.
---
### B. Monte Carlo Dropout
- Keep dropout **on** at test time.
- Run the same input through the network KKK times → get different predictions.
- Mean = prediction, variance = uncertainty.
- Cheap and easy to add to any network.

```python
model.train()  # keep dropout active 
preds = [model(x) for _ in range(50)] 
mean = torch.stack(preds).mean(0) 
std  = torch.stack(preds).std(0)
```

---
### C. Deep Ensembles
- Train **multiple models** (different initializations / bootstrapped data).
- At inference: predict with all models → measure variance across predictions.
- Empirically very strong baseline for uncertainty.
- More expensive than single model.
    
---
###  D. Predictive Distribution (Aleatoric)
- Modify last layer to output **mean and variance** (e.g., Gaussian likelihood).
- Train by maximizing log-likelihood.
- Variance gives direct aleatoric uncertainty.
```python
# final layer outputs 
mu, log_sigma loss = ((y - mu)**2 / (2* sigma**2)) + log_sigma
```

# Coding assistance
**Question: What's happening in this error "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."**

### Why you see this error

In your **PDE loss** you’re doing:
```python
u_t = torch.autograd.grad(torch.sum(u_collocation), wave_data.t_collocation,                           create_graph=True, retain_graph=True)[0]  

u_tt = torch.autograd.grad(torch.sum(u_t), wave_data.t_collocation,                            create_graph=True, retain_graph=True)[0]
```
    
    Then later again you call:
```python
    u_x = torch.autograd.grad(torch.sum(u_collocation), wave_data.x_collocation,                           create_graph=True, retain_graph=True)[0]
    
    u_xx = torch.autograd.grad(torch.sum(u_x), wave_data.x_collocation,                            create_graph=True)[0]
```
    
- Here, you’re “branching” multiple times from the same graph (`u_collocation`) without retaining it.
- By default, PyTorch frees intermediate results after the first `autograd.grad()` or `backward()` to save memory.
That’s why the error says: _“trying to backward through the graph a second time”_.

# Debugging
**Question: Can you tell me if there's anything wrong with this code? -Code snippets-**

**Bug in complex reconstruction**
`torch.abs(result[mask,0] + result[mask,0] * 1j)`
- You duplicated `result[mask,0]` for both real & imag part.
- Should be `result[mask,0] + 1j*result[mask,1]`.

# Novel Ideas Brainstorming

**Question: Can you suggest some new ideas to measure uncertainty of PINN?**
*As of now, ChatGPT are unable to come up with new ideas, below suggestion are something that has already been explored but still have interesting ideas regardless.
### 1. **Bayesian Approaches**

- **Bayesian PINNs (BPINNs):** Place priors over network weights and infer posterior distributions (via variational inference, MCMC, or Laplace approximations). The variance in predictions gives epistemic uncertainty.
- **Hierarchical Priors:** Instead of fixed Gaussian priors, use hyperpriors that adapt to regions of high/low data density.
### 2. **Ensemble-Based Methods**

- **Deep Ensembles of PINNs:** Train multiple PINNs with different initializations / bootstrap samples. Disagreement among them measures uncertainty.
- **Snapshot Ensembles:** Use checkpoints from different stages of training as an ensemble (cheaper to train than many independent PINNs).
### 3. **Aleatoric vs Epistemic Separation**

- Add a **noise head** to the PINN that explicitly models observation noise as a learnable variance parameter.
- Compare variance in the PDE residual vs variance in data-fitting loss: residual variance could indicate epistemic uncertainty due to model inadequacy.

# Intuition to difficult concept

**Question: I understand that in MCMC method, we have to sample from posterior distribution. But why are we updating the parameter when doing so?**

**Intuition in one line:**
Updating θ in MCMC isn’t optimization — it’s simulating a _random walk in parameter space_ that spends time in each region proportional to its posterior probability.

# Bad responses

**Question: How do you resolve doing backward propagation 2 times with respect to the inputs?**

ChatGPT Solution: Clone the input to prevent it. 
*In general might not be a good idea to clone before doing Auto-differentiation since it will introduce more memory and compute overhead.
```python
x_high = schrodinger_data.x_high_boundary.clone().detach().requires_grad_(True)
t_high = schrodinger_data.t_high_boundary.clone().detach().requires_grad_(True)
```

