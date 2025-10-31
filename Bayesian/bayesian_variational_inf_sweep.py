import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import os
import logging
from pinn.lib import SchrodingerModel, SchrodingerData, Util, get_loss
import wandb
import math
from pinn.train_bayesian_var_inference import train

run = wandb.init(
        reinit="finish_previous",
        entity="viriyadhika1",
        project="pinn-lab1",
        name="Bayesian PINN"
)

# Sweep code
config = wandb.config

if __name__ == '__main__':
    train(wandb_run=run,     
        clip_norm = config.clip_norm,
        beta_scaling = config.beta_scaling,
        lr = config.lr,
        mc = config.mc,
        prior_std = config.prior_std,
        epochs = 10000
    )
