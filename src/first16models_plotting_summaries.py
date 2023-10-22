import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

independent_ens_60 = pd.read_csv(DATA_PATH_OUTPUT / 'optimal_to60 models_ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 60 Autoencoders

sequential_60 = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders

fullsequential_60 = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders

# Averaging AUC scores of all datasets for each autoencoder model in ensemble
ind_16_mean = np.array(round(independent_ens_60.iloc[:11, 1:].mean(axis=1), 4))
ind_60_mean = np.array(round(independent_ens_60.iloc[:, 1:].mean(axis=1), 4))

seq_16_mean = np.array(round(sequential_60.iloc[:11, 1:].mean(axis=1), 4))
seq_60_mean = np.array(round(sequential_60.iloc[:, 1:].mean(axis=1), 4))

fullseq_16_mean = np.array(round(fullsequential_60.iloc[:11, 1:].mean(axis=1), 4))
fullseq_60_mean = np.array(round(fullsequential_60.iloc[:, 1:].mean(axis=1), 4))


# plotting
# summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
summary_16models = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
summary_16models['Number of Base Models'] = [i for i in range(1,12)]
summary_16models['Sequential Autoencoder Ensemble'] = seq_16_mean
summary_16models['Full Sequential Autoencoder Ensemble'] = fullseq_16_mean
summary_16models['Independent Autoencoder Ensemble'] = ind_16_mean

# ploting the 
sns.lineplot(data = summary_16models[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble']])
plt.legend(loc='best')
plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
plt.xlabel('Number of autoencoder models')
plt.ylabel('Mean of AUC scores')
plt.savefig(DATA_PATH_OUTPUT / 'graphics' /'summary_10models.jpg')
plt.show()