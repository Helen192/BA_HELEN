import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

# get scores of all experiments related to full sequential ensembles
to16_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders
to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0)
ens_offullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'Ensemble_OfFullSequential.csv')  #an ensemble of 10 different sequential models, in which each sequential model contains 10 different Autoencoders
unoptimal_to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'unoptimal_60models_full_sequential.csv', index_col=0) # This are results of using unoptimal auto with unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=1000



# get max score of each dataset trained by different sequential ensembles
to16_fullsequential_max = np.array(round(to16_fullsequential.iloc[:, 1:].max(axis=0), 4))
to60_fullsequential_max = np.array(round(to60_fullsequential.iloc[:, 1:].max(axis=0), 4))
ens_offullsequential_max = np.array(round(ens_offullsequential.loc[:, ['cardio mean', 'gas-drift mean', 'satellite mean', 'magic_telescope mean', 'pendigits mean', 'phoneme mean', 'pollen mean', 'speech mean', 'waveform mean']].max(axis=0), 4))
unoptimal_to60_fullsequential_max = np.array(round(unoptimal_to60_fullsequential.iloc[:, 1:].max(axis=0), 4))


fullsequential_df = pd.DataFrame(columns=['dataset', 'fullsequential_16', 'fullsequential_60','unoptimal_fullsequential','ensemble_10FullSequentials'])
fullsequential_df['dataset'] = LEGEND
fullsequential_df['fullsequential_16'] = to16_fullsequential_max
fullsequential_df['fullsequential_60'] = to60_fullsequential_max
fullsequential_df['ensemble_10FullSequentials'] = ens_offullsequential_max
fullsequential_df['unoptimal_fullsequential'] = unoptimal_to60_fullsequential_max

# save result to .csv and export
#fullsequential_df.to_csv(DATA_PATH_OUTPUT / 'summary_fullsequential_ensemble_results.csv')
print(fullsequential_df)

#print the best AUC score of each dataset trained by full sequential ensemble models
print(fullsequential_df.iloc[:, 1:].max(axis=1))
# print the average 
# the best AUC score of each dataset trained by full sequential ensemble models
fullsequential_df_max = fullsequential_df.iloc[:, 1:].max(axis=1)
# the average of best AUC score of all datasets
fullsequential_mean = round(np.mean(fullsequential_df_max), 4)
unoptimal_fullseq_mean = round(np.mean(unoptimal_to60_fullsequential_max), 4)
to60_full_mean = round(np.mean(fullsequential_df['fullsequential_60']), 4)
print(to60_full_mean)
print(f"optimal: {fullsequential_mean} vs. unoptimal: {unoptimal_fullseq_mean}")
