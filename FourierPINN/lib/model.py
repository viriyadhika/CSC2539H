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

class MultiscaleFourierEmbedding(nn.Module):
    def __init__(self, in_dim: int, num_features: int, sigmas: list[float]):
        """
        Args:
            in_dim: input dimension (e.g. 2 for (x,t))
            num_features: number of random features per block
            sigmas: list of scales (one per block), e.g. [1.0, 10.0, 50.0]
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_features = num_features
        self.sigmas = sigmas
        self.M = len(sigmas)

        # For each sigma, create a random projection matrix B^i
        # Shape: (M, num_features, in_dim)
        B = torch.randn(self.M, num_features, in_dim)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim)
        returns: (batch, M * 2 * num_features)
        """
        batch_size = x.shape[0]

        feats = []
        for i, sigma in enumerate(self.sigmas):
            proj = (x @ self.B[i].T) * sigma  # (batch, num_features)
            cos_feat = torch.cos(2 * torch.pi * proj)
            sin_feat = torch.sin(2 * torch.pi * proj)
            feats.append(torch.cat([cos_feat, sin_feat], dim=-1))

        return torch.cat(feats, dim=-1)  # (batch, M * 2 * num_features)

class FFMLP(nn.Module):
    def __init__(self, sigmas: list[float]) -> None:
        super().__init__()
        self.embedding = MultiscaleFourierEmbedding(in_dim=2, num_features=16, sigmas=sigmas)
        self.ff = nn.Sequential(
            nn.Linear(96, 200),
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
    
