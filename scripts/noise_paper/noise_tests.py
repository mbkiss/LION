# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Max Kiss, Ander Biguri
# =============================================================================

#%% 0 - Imports
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim

# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# LION imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_learned_denoising_experiments as ct_denoising

# Just a temporary SSIM that takes torch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


#%% 1 - Settings
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Device
device = torch.device("cuda:1")
torch.cuda.set_device(device)

#%% 2 - Define experiment
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all the experiments we need to run for the noise paper

# Experimental noisy dataset
experiment = ct_denoising.ExperimentalNoiseDenoising()


#%% 3 - Obtaining Datasets from experiments
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ExpNoise_data = experiment.get_training_dataset()

#print(experiment.experiment_params.flat_field_correction)

##############################################################
# REMOVE THIS CHUNK IN THE FINAL VERSION
indices = torch.arange(100)
ExpNoise_data = data_utils.Subset(ExpNoise_data, indices)


# REMOVE THIS CHUNK IN THE FINAL VERSION
##############################################################

#%% 4 - Define Data Loader
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is standard pytorch, no LION here.

batch_size = 1
ExpNoise_dataloader = DataLoader(ExpNoise_data, batch_size, shuffle=False)


# Simulated noisy dataset
experiment = ct_denoising.ArtificialNoiseDenoising()
ArtNoise_data = experiment.get_training_dataset()

##############################################################
# REMOVE THIS CHUNK IN THE FINAL VERSION
indices = torch.arange(100)
ArtNoise_data = data_utils.Subset(ArtNoise_data, indices)
ArtNoise_dataloader = DataLoader(ArtNoise_data, batch_size, shuffle=False)

#MSE_metric = np.zeros_like(next(iter(ExpNoise_dataloader))[0])
MSE_metric = np.zeros(100)


for i, (sino, target) in enumerate(ExpNoise_dataloader):
	
	sino1, target1 = sino, target #ExpNoise_dataloader[i]
	sino2, target2 = next(iter(ArtNoise_dataloader))
	print("Mean of sino1:", torch.mean(sino1), "Mean of target1:", torch.mean(target1))
	
	MSE = nn.MSELoss()
	#print(MSE_metric.shape)
	MSE_metric[i] = MSE(sino1,target1)
	#print(MSE(sino2,sino1))

print(MSE_metric)	
