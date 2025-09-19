import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import os
import logging
import wandb

class MultiScaleFourierEmbedding(nn.Module):
    def __init__(self, n_input: int, n_out: int, sigmas: list[float]) -> None:
        super().__init__()
        Bs = [] 
        for sigma in sigmas:
            Bs.append(torch.randn([n_input, n_out // 2]) ** sigma)
        self.register_buffer("Bs", torch.stack(Bs, dim=0))

    # X = n_batch x n_input
    def forward(self, X):
        Hs = []
        for B in self.Bs:
            out = X @ B
            Hs.append(torch.cat([torch.sin(out), torch.cos(out)], dim=1))

        return Hs
    
class MFFFourier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sigmas = [1., 10.]
        n_hidden = 200

        self.multi_scale_ff = MultiScaleFourierEmbedding(2, n_hidden, sigmas)
        self.ff = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
        )
        self.last_layer = nn.Linear(len(sigmas) * n_hidden, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Apply Xavier initialization to linear layers"""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization (also called Glorot uniform)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> None:
        X = torch.stack([x, t], dim=1)
        embeddings = self.multi_scale_ff(X)
        result = []
        for embedding in embeddings:
            # Use the same weight but evolve independently
            result.append(self.ff(embedding))

        # Cat it together
        penultimate_layer = torch.cat(result, dim=1)
        
        return self.last_layer(penultimate_layer)

class AnisotropicFourierEmbedding(nn.Module):
    def __init__(self, num_features: int, sigmas: list[float]):
        """
        Args:
            num_features: number of random Fourier features
            sigmas: list of per-dimension scales, e.g. [1.0, 10.0] for (x,t)
        """
        super().__init__()
        self.num_features = num_features
        self.sigmas = torch.tensor(sigmas)  # (in_dim,)
        self.in_dim = len(sigmas)

        # Random projection matrix: (num_features, in_dim)
        B: torch.Tensor = torch.randn(num_features, self.in_dim)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim), e.g. (batch, 2) for (x,t)
        returns: (batch, 2*num_features)
        """
        # Scale each dimension independently
        x_scaled = x * self.sigmas.to(x.device)   # (batch, in_dim)

        # Project
        proj = x_scaled @ self.B.T  # (batch, num_features)

        # Fourier features
        cos_feat = torch.cos(2 * torch.pi * proj)
        sin_feat = torch.sin(2 * torch.pi * proj)
        return torch.cat([cos_feat, sin_feat], dim=-1)


class AnisotropicFourierMLP(nn.Module):
    def __init__(self, sigmas: list[float]) -> None:
        super().__init__()
        self.embedding = AnisotropicFourierEmbedding(num_features=16, sigmas=sigmas)
        self.ff = nn.Sequential(
            nn.Linear(32, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Apply Xavier initialization to linear layers"""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization (also called Glorot uniform)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        X = torch.stack([x, t], dim=1)
        y = self.embedding(X)
        y = self.ff(y)
        return y


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Apply Xavier initialization to linear layers"""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization (also called Glorot uniform)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        X = torch.stack([x, t], dim=1)
        return self.ff(X)
    
