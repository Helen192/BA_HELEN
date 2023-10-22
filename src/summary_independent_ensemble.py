import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

# get scores of all experiments related to independent ensemble model
inspect_ens = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_ensemble_auc.csv', index_col=0)  # the results of ensemble contains 6 Autoencoders
ens_num = pd.read_csv(DATA_PATH_OUTPUT / 'ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 16 Autoencoders
to60_ens = pd.read_csv(DATA_PATH_OUTPUT / 'to60 models_ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 60 Autoencoders
unoptimal_300epochs_inde_ens = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batch_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)   #LATENT_LIST_Version2 = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=300, batch=100
ver2_optimal = pd.read_csv(DATA_PATH_OUTPUT / 'optimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
ind_ens_200models = pd.read_csv(DATA_PATH_OUTPUT / '200models_independent_ensemble.csv', index_col=0)

# get max score of each dataset trained by different ensembles
inspect_ens_max = np.array(round(inspect_ens.iloc[:, 1:].max(axis=0), 4))
ens_num_max = np.array(round(ens_num.iloc[:, 1:].max(axis=0), 4))
to60_ens_max = np.array(round(to60_ens.iloc[:, 1:].max(axis=0), 4))
unoptimal_300epochs_inde_ens_max = np.array(round(unoptimal_300epochs_inde_ens.iloc[-1:, 1:].max(axis=0), 4))
ver2_optimal_max = np.array(round(ver2_optimal.iloc[-1:, 1:].max(axis=0), 4))
to200_ens_max =  np.array(round(ind_ens_200models.iloc[:61, 1:].max(axis=0), 4))

independent_ensemble_df = pd.DataFrame(columns=['dataset', 'ensemble_6', 'ensemble_16','ensemble_60', 'ver2_optimal','unoptimal_300epochs', 'ensemble_200'])
independent_ensemble_df['dataset'] = LEGEND
independent_ensemble_df['ensemble_6'] = inspect_ens_max
independent_ensemble_df['ensemble_16'] = ens_num_max
independent_ensemble_df['ensemble_60'] = to60_ens_max
independent_ensemble_df['ver2_optimal'] = ver2_optimal_max
independent_ensemble_df['unoptimal_300epochs'] = unoptimal_300epochs_inde_ens_max
independent_ensemble_df['ensemble_200'] = to200_ens_max

# save result to .csv and export
#independent_ensemble_df.to_csv(DATA_PATH_OUTPUT / 'summary_independent_ensemble_results.csv')
print(independent_ensemble_df)

#print the best AUC score of each dataset trained by independent ensemble models
#print(independent_ensemble_df.iloc[:, 1:-1].max(axis=1))
 # print the average 
inde_ens_df_max = independent_ensemble_df.iloc[:, 1:-1].max(axis=1)
inde_ens_mean = round(np.mean(inde_ens_df_max), 4)
to60_ens_mean = round(np.mean(to60_ens_max), 4)
ver2_optimal_mean = round(np.mean(ver2_optimal_max), 4)
unoptimal_300epochs_inde_ens_mean = round(np.mean(unoptimal_300epochs_inde_ens_max), 4)
to200_ens_mean = round(np.mean(to200_ens_max), 4)
print(f"optimal max: {to60_ens_mean} vs Ver2_optimal {ver2_optimal_mean} vs unoptimal-300epochs: {unoptimal_300epochs_inde_ens_mean} vs 200models: {to200_ens_mean}")
#print(round(np.mean(to60_ens_max), 4))