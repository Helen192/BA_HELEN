import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

# 1. Autoencoder
# get scores of all experiments related to Autoencoder models
auto_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_autoencoder_auc.csv', index_col=0)  # optimal autoencoder: LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
unoptimal_auto = pd.read_csv(DATA_PATH_OUTPUT / 'not_finetun_inspect_autoencoder_auc.csv', index_col=0) # model uses: LATENT_LIST = [4, 32, 16, 4, 4, 4, 4, 64, 32], epoch=1000, epoch=1000, batch=64
ver2_unoptimal_auto = pd.read_csv(DATA_PATH_OUTPUT / 'not_finetun_version2_inspect_autoencoder_auc.csv', index_col=0) #LATENT_LIST_Version2 = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoc=1000, batch=64
unoptimal_300epochs = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batch_not_finetun_version2_inspect_autoencoder_auc.csv', index_col=0)  # #LATENT_LIST_Version2 = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=300, batch=100
ver3_auto = pd.read_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_autoencoder_500e_64b_auc.csv', index_col=0) # unoptimal_latent_ver3 = [32, 64, 32, 8, 8, 4, 4, 128, 64], epoch=500, batch=64

autoencoder_max = round(auto_auc.iloc[:, 2],4)
unoptimal_auto_max = round(unoptimal_auto.iloc[:, 2],4)
ver2_unoptimal_auto_max = round(ver2_unoptimal_auto.iloc[:, 2],4)
unoptimal_300epochs_max = round(unoptimal_300epochs.iloc[:, 2],4)
ver3_auto_max = round(ver3_auto.iloc[:, 2:], 4)

# get mean values 
autoencoder_mean = round(np.mean(autoencoder_max), 4)
unoptimal_auto_mean = round(np.mean(unoptimal_auto_max), 4)
ver2_unoptimal_auto_mean = round(np.mean(ver2_unoptimal_auto_max), 4)
unoptimal_300epochs_mean = round(np.mean(unoptimal_300epochs_max), 4)
ver3_auto_mean = round(np.mean(ver3_auto_max), 4)


#2. Sequential Ensemble
# get scores of all experiments related to sequential ensembles
#to16_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders, optimal model - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
to60_sequential = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0) # to60 base models, optimal models - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
unoptimal_to60_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'ver1_unoptimal_60models_sequential.csv', index_col=0) # model uses: LATENT_LIST = [4, 32, 16, 4, 4, 4, 4, 64, 32], epoch=1000, epoch=1000, batch=64
ver2_unoptimal_to60_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'unoptimal_60models_sequential.csv', index_col=0) # This are results of using unoptimal auto with unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=1000, batch=64
unoptimal_300epochs_sequential = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batch_unoptimal_60models_sequential.csv', index_col=0) # unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epochs=300, batch=100
ver3_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_sequential_500e_64b.csv', index_col=0)

# get max score of each dataset trained by different sequential ensembles
#to16_sequential_max = np.array(round(to16_sequential.iloc[:, 1:].max(axis=0), 4))
to60_sequential_max = np.array(round(to60_sequential.iloc[:, 1:].max(axis=0), 4))
unoptimal_to60_sequential_max = np.array(round(unoptimal_to60_sequential.iloc[:, 1:].max(axis=0), 4))
ver2_unoptimal_to60_sequential_max = np.array(round(ver2_unoptimal_to60_sequential.iloc[:, 1:].max(axis=0), 4))
unoptimal_300epochs_sequential_max = np.array(round(unoptimal_300epochs_sequential.iloc[:, 1:].max(axis=0), 4))
ver3_sequential_max = np.array(round(ver3_sequential.iloc[:, 1:].max(axis=0), 4))

# get mean values
#to16_sequential_mean = round(np.mean(to16_sequential_max, 4))
to60_sequential_mean = round(np.mean(to60_sequential_max), 4)
unoptimal_to60_sequential_mean = round(np.mean(unoptimal_to60_sequential_max), 4)
ver2_unoptimal_to60_sequential_mean = round(np.mean(ver2_unoptimal_to60_sequential_max), 4)
unoptimal_300epochs_sequential_mean = round(np.mean(unoptimal_300epochs_sequential_max), 4)
ver3_sequential_mean = round(np.mean(ver3_sequential_max), 4)


#3. Full Sequential
# get scores of all experiments related to full sequential ensembles
#to16_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # to 60 models, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
unoptimal_to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'ver1_unoptimal_60models_full_sequential.csv', index_col=0) # model uses: LATENT_LIST = [4, 32, 16, 4, 4, 4, 4, 64, 32], epoch=1000, epoch=1000, batch=64
ver2_unoptimal_to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'unoptimal_60models_full_sequential.csv', index_col=0) #  unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=1000, batch=64
unoptimal_300epochs_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batchunoptimal_60models_full_sequential.csv', index_col=0) # unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epochs=300, batch=100
ver3_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_full_sequential_500e_64b.csv', index_col=0)

# get max score of each dataset trained by different sequential ensembles
#to16_fullsequential_max = np.array(round(to16_fullsequential.iloc[:, 1:].max(axis=0), 4))
to60_fullsequential_max = np.array(round(to60_fullsequential.iloc[:, 1:].max(axis=0), 4))
unoptimal_to60_fullsequential_max = np.array(round(unoptimal_to60_fullsequential.iloc[:, 1:].max(axis=0), 4))
ver2_unoptimal_to60_fullsequential_max = np.array(round(ver2_unoptimal_to60_fullsequential.iloc[:, 1:].max(axis=0), 4))
unoptimal_300epochs_fullsequential_max = np.array(round(unoptimal_300epochs_fullsequential.iloc[:, 1:].max(axis=0), 4))
ver3_fullsequential_max = np.array(round(ver3_fullsequential.iloc[:, 1:].max(axis=0), 4))

#get mean values
#to16_fullsequential_mean = round(np.mean(to16_fullsequential_max, 4))
to60_fullsequential_mean = round(np.mean(to60_fullsequential_max), 4)
unoptimal_to60_fullsequential_mean = round(np.mean(unoptimal_to60_fullsequential_max), 4)
ver2_unoptimal_to60_fullsequential_mean = round(np.mean(ver2_unoptimal_to60_fullsequential_max), 4)
unoptimal_300epochs_fullsequential_mean = round(np.mean(unoptimal_300epochs_fullsequential_max), 4)
ver3_fullsequential_mean = round(np.mean(ver3_fullsequential_max), 4)


#4. Independent Ensemble
# get scores of all experiments related to independent ensemble model
inspect_ens = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_ensemble_auc.csv', index_col=0)  # the results of ensemble contains 6 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
ens_num = pd.read_csv(DATA_PATH_OUTPUT / 'ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 16 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
to60_ens = pd.read_csv(DATA_PATH_OUTPUT / 'optimal_to60 models_ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 60 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
ver1_ens = pd.read_csv(DATA_PATH_OUTPUT / 'ver1_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
ver2_ens = pd.read_csv(DATA_PATH_OUTPUT / 'ver2_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
unoptimal_ens = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batch_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
ver3_ens = pd.read_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_ind_ensemble_500e_64b.csv', index_col=0)


# get max score of each dataset trained by different ensembles
inspect_ens_max = np.array(round(inspect_ens.iloc[-1:, 1:].max(axis=0), 4))
ens_num_max = np.array(round(ens_num.iloc[-1:, 1:].max(axis=0), 4))
to60_ens_max = np.array(round(to60_ens.iloc[-1:, 1:].max(axis=0), 4))
ver1_ens_max = np.array(round(ver1_ens.iloc[-1:, 1:].max(axis=0), 4))
ver2_ens_max = np.array(round(ver2_ens.iloc[-1:, 1:].max(axis=0), 4))
unoptimal_ens_max = np.array(round(unoptimal_ens.iloc[-1:, 1:].max(axis=0), 4))
ver3_ens_max = np.array(round(ver3_ens.iloc[-1:, 1:].max(axis=0), 4))

# get mean score
to60_ens_mean = round(np.mean(to60_ens_max), 4)
ver1_ens_mean = round(np.mean(ver1_ens_max), 4)
ver2_ens_mean = round(np.mean(ver2_ens_max), 4)
unoptimal_ens_mean = round(np.mean(unoptimal_ens_max), 4)
ver3_ens_mean = round(np.mean(ver3_ens_max), 4)


# 5. Create DataFrame
summary_df = pd.DataFrame(columns=['Type of base model', 'Autoencoder', 'Independent Ensemble','Sequential Ensemble', 'Full Sequential Ensemble'])
summary_df['Type of base model'] = ['Base 1', 'Base 2', 'Base 3', 'Base 4', 'Base 5']
summary_df['Autoencoder'] = [autoencoder_mean, unoptimal_auto_mean, ver2_unoptimal_auto_mean, unoptimal_300epochs_mean, ver3_auto_mean]  # thieu unoptimal_auto_mean
summary_df['Independent Ensemble'] = [to60_ens_mean, ver1_ens_mean, ver2_ens_mean, unoptimal_ens_mean,ver3_ens_mean]
summary_df['Sequential Ensemble'] = [to60_sequential_mean, unoptimal_to60_sequential_mean, ver2_unoptimal_to60_sequential_mean, unoptimal_300epochs_sequential_mean, ver3_sequential_mean]
summary_df['Full Sequential Ensemble'] = [to60_fullsequential_mean, unoptimal_to60_fullsequential_mean, ver2_unoptimal_to60_fullsequential_mean, unoptimal_300epochs_fullsequential_mean, ver3_fullsequential_mean]

#print(summary_df)

# 6. Plot
auto_df = [autoencoder_mean, unoptimal_auto_mean, ver2_unoptimal_auto_mean, unoptimal_300epochs_mean, ver3_auto_mean]
ensemble_df = [to60_ens_mean, ver1_ens_mean, ver2_ens_mean, unoptimal_ens_mean,ver3_ens_mean]
sequential_df = [to60_sequential_mean, unoptimal_to60_sequential_mean, ver2_unoptimal_to60_sequential_mean, unoptimal_300epochs_sequential_mean, ver3_sequential_mean]
fullsequential_df = [to60_fullsequential_mean, unoptimal_to60_fullsequential_mean, ver2_unoptimal_to60_fullsequential_mean, unoptimal_300epochs_fullsequential_mean, ver3_fullsequential_mean]


sns.set_palette('pastel')
w=0.15
x = ['Base 1', 'Base 2', 'Base 3', 'Base 4', 'Base 5']
bar1 = np.arange(len(x))
bar2 = [i+w for i in bar1]
bar3 = [i + w for i in bar2]
bar4 = [i + w for i in bar3]

plt.bar(bar1, auto_df, w, label='Autoencoders')
plt.bar(bar2, ensemble_df, w, label='Independent Autoencoder Ensemble')
plt.bar(bar3, sequential_df, w, label='Sequential Autoencoder Ensembles')
plt.bar(bar4, fullsequential_df, w, label='Full Sequential Autoencoder Ensembles')

plt.xlabel('Type of base learners')
plt.ylabel('The mean AUC scores')
plt.title('The mean AUC scores of each type of model using various base learners')
plt.ylim(0.75, 0.9)
plt.xticks(bar2, x)
plt.legend(loc='best')
plt.savefig(DATA_PATH_OUTPUT / 'graphics' / 'seq_fullseq_barplot_summary.jpg')
plt.show()