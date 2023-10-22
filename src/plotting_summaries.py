import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

independent_ens_16 = pd.read_csv(DATA_PATH_OUTPUT / 'ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 16 Autoencoders
independent_ens_60 = pd.read_csv(DATA_PATH_OUTPUT / 'optimal_to60 models_ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 60 Autoencoders

sequential_16 = pd.read_csv(DATA_PATH_OUTPUT / 'sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders
sequential_60 = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders
rand12_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'rand1234_60models_sequential.csv', index_col=0)
rand75_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'rand75300_60models_sequential.csv', index_col=0)
rand11_sequential = pd.read_csv(DATA_PATH_OUTPUT / 'rand11111_60models_sequential.csv', index_col=0)

fullsequential_16 = pd.read_csv(DATA_PATH_OUTPUT / 'full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders
fullsequential_60 = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders
rand12_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'rand1234_60models_fullsequential.csv', index_col=0)
rand75_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'rand75300_60models_fullsequential.csv', index_col=0)
rand11_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'rand11111_60models_fullsequential.csv', index_col=0)


ind_ens_200models = pd.read_csv(DATA_PATH_OUTPUT / '200models_independent_ensemble.csv', index_col=0)
# Averaging AUC scores of all datasets for each autoencoder model in ensemble
ind_16_mean = np.array(round(independent_ens_16.iloc[:, 1:].mean(axis=1), 4))
ind_60_mean = np.array(round(independent_ens_60.iloc[:, 1:].mean(axis=1), 4))
ind_ens_mean = np.array(round(ind_ens_200models.iloc[:60, 1:].mean(axis=1), 4))

# synchronizing
syn = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'synchronizing_ver2.csv', index_col=0)
syn_mean = np.array(round(syn.iloc[:, 1:].mean(axis=1), 4))
#plt.plot([i for i in range(1, 6)], syn_mean, label='Synchronizing Sequential Autoencoder Ensemble')
#plt.legend(loc='best')
#plt.title('Correlation between number of autoencoders and mean of AUC scores \nin a synchronizing sequential autoencoder ensemble', fontsize=10)
#plt.xlabel('Number of autoencoder models')
#plt.ylabel('AUC scores')
#plt.savefig(DATA_PATH_OUTPUT / 'synchronizing' / 'synchronizing_plotting.jpg')
#plt.show()

seq_16_mean = np.array(round(sequential_16.iloc[:, 1:].mean(axis=1), 4))
seq_60_mean = np.array(round(sequential_60.iloc[:, 1:].mean(axis=1), 4))
rand12_sequential_mean = np.array(round(rand12_sequential.iloc[:, 1:].mean(axis=1), 4))
rand75_sequential_mean = np.array(round(rand75_sequential.iloc[:, 1:].mean(axis=1), 4))
rand11_sequential_mean = np.array(round(rand11_sequential.iloc[:, 1:].mean(axis=1), 4))

fullseq_16_mean = np.array(round(fullsequential_16.iloc[:, 1:].mean(axis=1), 4))
fullseq_60_mean = np.array(round(fullsequential_60.iloc[:, 1:].mean(axis=1), 4))
rand12_fullsequential_mean = np.array(round(rand12_fullsequential.iloc[:, 1:].mean(axis=1), 4))
rand75_fullsequential_mean = np.array(round(rand75_fullsequential.iloc[:, 1:].mean(axis=1), 4))
rand11_fullsequential_mean = np.array(round(rand11_fullsequential.iloc[:, 1:].mean(axis=1), 4))



# Calculate the average of 3 different means values
average_seq = (seq_60_mean + rand12_sequential_mean + rand75_sequential_mean + rand11_sequential_mean) / 4
average_full = (fullseq_60_mean + rand12_fullsequential_mean + rand75_fullsequential_mean + rand11_fullsequential_mean) / 4
average_ind = (seq_60_mean + ind_ens_mean) / 2


# plotting average
# summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
#average_summary = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
#average_summary['Number of Base Models'] = [i for i in range(1, 61)]
#average_summary['Sequential Autoencoder Ensemble'] = average_seq
#average_summary['Full Sequential Autoencoder Ensemble'] = average_full
#average_summary['Independent Autoencoder Ensemble'] = average_ind
#
#
#sns.lineplot(data = average_summary[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble']])
##plt.plot(ind_axis_x, ind_60_mean)
##plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
#plt.legend(loc='best')
#plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
#plt.xlabel('Number of autoencoder models')
#plt.ylabel('Mean of AUC scores')
#plt.savefig(DATA_PATH_OUTPUT / 'plott200' /'average_summary_60models.jpg')
#plt.show()
#
#

# plotting
# summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
summary_60models['Number of Base Models'] = [i for i in range(1, 61)]
summary_60models['Sequential Autoencoder Ensemble'] = seq_60_mean
summary_60models['Full Sequential Autoencoder Ensemble'] = fullseq_60_mean
summary_60models['Independent Autoencoder Ensemble'] = ind_60_mean

# ploting the 
# palette = sns.color_palette("mako_r", 3)
# sns.lineplot(data = summary_60models[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble']], palette = palette)
# #plt.plot(ind_axis_x, ind_60_mean)
# #plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
# plt.legend(loc='best')
# plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
# plt.xlabel('Number of autoencoder models')
# plt.ylabel('Mean of AUC scores')
# plt.savefig(DATA_PATH_OUTPUT / 'graphics' / 'summary_60models.jpg')
# plt.show()


# plotting for rand 1234
summary_rand12 = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
summary_rand12['Number of Base Models'] = [i for i in range(1, 61)]
summary_rand12['Sequential Autoencoder Ensemble'] = rand12_sequential_mean
summary_rand12['Full Sequential Autoencoder Ensemble'] = rand12_fullsequential_mean
summary_rand12['Independent Autoencoder Ensemble'] = ind_60_mean

#palette = sns.color_palette("mako_r", 2)
#sns.lineplot(data = summary_rand12[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble']], palette = palette)
##plt.plot(ind_axis_x, ind_60_mean)
##plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
#plt.legend(loc='best')
#plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
#plt.xlabel('Number of autoencoder models')
#plt.ylabel('Mean of AUC scores')
#plt.savefig(DATA_PATH_OUTPUT / 'graphics' / 'rand1234_summary_60models.jpg')
#plt.show()

# # plotting for rand 75300
summary_rand75 = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble'])
summary_rand75['Number of Base Models'] = [i for i in range(1, 61)]
summary_rand75['Sequential Autoencoder Ensemble'] = rand75_sequential_mean
summary_rand75['Full Sequential Autoencoder Ensemble'] = rand75_fullsequential_mean

#palette = sns.color_palette("mako_r", 2)
#sns.lineplot(data = summary_rand75[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble']], palette = palette)
##plt.plot(ind_axis_x, ind_60_mean)
##plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
#plt.legend(loc='best')
#plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
#plt.xlabel('Number of autoencoder models')
#plt.ylabel('Mean of AUC scores')
#plt.savefig(DATA_PATH_OUTPUT / 'graphics' / 'rand75300_summary_60models.jpg')
#plt.show()


# # plotting for rand 11111
# summary_rand11 = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble'])
# summary_rand11['Number of Base Models'] = [i for i in range(1, 61)]
# summary_rand11['Sequential Autoencoder Ensemble'] = rand11_sequential_mean
# summary_rand11['Full Sequential Autoencoder Ensemble'] = rand11_fullsequential_mean
# 
# palette = sns.color_palette("mako_r", 2)
# sns.lineplot(data = summary_rand11[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble']], palette = palette)
# #plt.plot(ind_axis_x, ind_60_mean)
# #plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
# plt.legend(loc='best')
# plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
# plt.xlabel('Number of autoencoder models')
# plt.ylabel('Mean of AUC scores')
# plt.savefig(DATA_PATH_OUTPUT / 'graphics' / 'rand11111_summary_60models.jpg')
# plt.show()


seq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_sequential.csv', index_col=0)
fullseq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_fullsequential.csv', index_col=0)
ind_ens_200models = pd.read_csv(DATA_PATH_OUTPUT / '200models_independent_ensemble.csv', index_col=0)
seq_200models_mean = np.array(round(seq_200models.iloc[:5, 1:].mean(axis=1), 4))
fullseq_200models_mean = np.array(round(fullseq_200models.iloc[:5, 1:].mean(axis=1), 4))
ind_ens_200models_mean = np.array(round(ind_ens_200models.iloc[:5, 1:].mean(axis=1), 4))
# 
# plotting 200 models
summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
summary_200models = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble', 'Synchronizing Sequential Autoencoder Ensemble'])
summary_200models['Number of Base Models'] = [i for i in range(1, 6)]
summary_200models['Sequential Autoencoder Ensemble'] = seq_200models_mean
summary_200models['Full Sequential Autoencoder Ensemble'] = fullseq_200models_mean
summary_200models['Independent Autoencoder Ensemble'] = ind_ens_200models_mean
summary_200models['Synchronizing Sequential Autoencoder Ensemble'] = syn_mean
# 
#  
# 
sns.lineplot(data = summary_200models[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble', 'Synchronizing Sequential Autoencoder Ensemble']])
#plt.plot(ind_axis_x, ind_60_mean)
#plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
plt.legend(loc='best')
plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
plt.xlabel('Number of autoencoder models')
plt.ylabel('Mean of AUC scores')
plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
plt.savefig(DATA_PATH_OUTPUT / 'synchronizing' / 'fullsummary_5models.jpg')
plt.show()

