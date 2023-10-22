import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

# get scores of all experiments related to sequential ensembles
to16_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders
to60_sequential = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0)
ens_ofsequential = pd.read_csv(DATA_PATH_OUTPUT / 'Ensemble_OfSequential.csv')  #an ensemble of 10 different sequential models, in which each sequential model contains 10 different Autoencoders
#new_to60_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'new_version_60models_sequential.csv', index_col=0) # This are the results of sequential, after adding some weights for prediction errors (follow Simon's method)
unoptimal_to60_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'unoptimal_60models_sequential.csv', index_col=0) # This are results of using unoptimal auto with unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=1000
seq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_sequential.csv', index_col=0)

# get max score of each dataset trained by different sequential ensembles
to16_sequential_max = np.array(round(to16_sequential.iloc[:, 1:].max(axis=0), 4))
to60_sequential_max = np.array(round(to60_sequential.iloc[:, 1:].max(axis=0), 4))
ens_ofsequential_max = np.array(round(ens_ofsequential.loc[:, ['cardio mean', 'gas-drift mean', 'satellite mean', 'magic_telescope mean', 'pendigits mean', 'phoneme mean', 'pollen mean', 'speech mean', 'waveform mean']].max(axis=0), 4))
#new_to60_sequential_max = np.array(round(new_to60_sequential.iloc[:, 1:].max(axis=0), 4))
unoptimal_to60_sequential_max = np.array(round(unoptimal_to60_sequential.iloc[:, 1:].max(axis=0), 4))
seq_200_max =  np.array(round(seq_200models.iloc[:61, 1:].max(axis=0), 4))


sequential_df = pd.DataFrame(columns=['dataset', 'sequential_16', 'sequential_60', 'unoptimal_sequential','ensemble_10Sequentials', 'sequential_200'])
sequential_df['dataset'] = LEGEND
sequential_df['sequential_16'] = to16_sequential_max
sequential_df['sequential_60'] = to60_sequential_max
#sequential_df['new_sequential_60'] = new_to60_sequential_max
sequential_df['ensemble_10Sequentials'] = ens_ofsequential_max
sequential_df['unoptimal_sequential'] = unoptimal_to60_sequential_max
sequential_df['sequential_200'] = seq_200_max

# save result to .csv and export
#sequential_df.to_csv(DATA_PATH_OUTPUT / 'summary_sequential_ensemble_results.csv')
print(sequential_df)

#print the best AUC score of each dataset trained by sequential ensemble models
print(sequential_df.iloc[:, 1:].max(axis=1))
# print the average 
# the best AUC score of each dataset trained by sequential ensemble models
sequential_df_max = sequential_df.iloc[:, 1:].max(axis=1)
## the average of best AUC score of all datasets
sequential_mean = round(np.mean(sequential_df_max), 4)
unoptimal_mean = round(np.mean(unoptimal_to60_sequential_max), 4)
seq_200_mean = round(np.mean(seq_200_max), 4)
print(round(np.mean(to60_sequential_max), 4))
print(f"optimal: {sequential_mean} vs. unoptimal: {unoptimal_mean} vs. 200models: {seq_200_mean}")

