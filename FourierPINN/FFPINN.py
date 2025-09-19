# %%
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

from lib.model import MFFFourier
from lib.lib import WaveData, WaveRawData, Util, device, PINNLoss, training_loop

if __name__ == '__main__':
    wave_raw_data = WaveRawData()
    util = Util()
    wave_data = WaveData(wave_raw_data, 360, 360, 360, util=util)
    # %%
    mff_fourier = MFFFourier().to(device)

    optimizer = torch.optim.Adam(mff_fourier.parameters())
    criterion = PINNLoss()

    # %%
    optimizer = torch.optim.Adam(mff_fourier.parameters())

    # %%
    training_loop(epochs=40000, wave_data=wave_data, criterion=criterion, model=mff_fourier, optimizer=optimizer)

    # %%
    x, t = torch.tensor(wave_raw_data.X.reshape(-1), dtype=torch.float, device=device), torch.tensor(wave_raw_data.T.reshape(-1), dtype=torch.float, device=device)
    y = mff_fourier(x, t)

