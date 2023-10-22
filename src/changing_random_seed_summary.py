import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

sequential_rand42 = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0) # LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64, random_seed = 42

fullsequential_rand42 = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64, random_seed=42

# Averaging AUC scores of all datasets for each autoencoder model in ensemble

seq_60_mean = np.array(round(sequential_rand42.iloc[:, 1:].mean(axis=1), 4))

fullseq_60_mean = np.array(round(fullsequential_rand42.iloc[:, 1:].mean(axis=1), 4))
