import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *



rand42_seq = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders
rand12_seq = pd.read_csv(DATA_PATH_OUTPUT / 'rand1234_60models_sequential.csv', index_col=0)
rand75_seq = pd.read_csv(DATA_PATH_OUTPUT / 'rand75300_60models_sequential.csv', index_col=0)
rand11_seq = pd.read_csv(DATA_PATH_OUTPUT / 'rand11111_60models_sequential.csv', index_col=0)

rand42_full = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders
rand12_full = pd.read_csv(DATA_PATH_OUTPUT / 'rand1234_60models_fullsequential.csv', index_col=0)
rand75_full = pd.read_csv(DATA_PATH_OUTPUT / 'rand75300_60models_fullsequential.csv', index_col=0)
rand11_full = pd.read_csv(DATA_PATH_OUTPUT / 'rand11111_60models_fullsequential.csv', index_col=0)

# Calculate max
rand42_seq_max = np.array(round(rand42_seq.iloc[:, 1:].max(axis=0), 4))
rand12_seq_max = np.array(round(rand12_seq.iloc[:, 1:].max(axis=0), 4))
rand75_seq_max = np.array(round(rand75_seq.iloc[:, 1:].max(axis=0), 4))
rand11_seq_max = np.array(round(rand11_seq.iloc[:, 1:].max(axis=0), 4))

sequential_df = pd.DataFrame(columns=['dataset', 'Rand42', 'Rand12','Rand75', 'Rand11'])
sequential_df['dataset'] = LEGEND
sequential_df['Rand42'] = rand42_seq_max
sequential_df['Rand12'] = rand12_seq_max
sequential_df['Rand75'] = rand75_seq_max
sequential_df['Rand11'] = rand11_seq_max




sequential_df_max = sequential_df.iloc[:, 1:-1].max(axis=1)
sequential_df_mean = round(np.mean(sequential_df_max), 4)
print(sequential_df)
print(sequential_df_max)
print(sequential_df_mean)