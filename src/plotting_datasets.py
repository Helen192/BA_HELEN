import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *


independent_ens_60 = pd.read_csv(DATA_PATH_OUTPUT / 'optimal_to60 models_ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 60 Autoencoders

sequential_60 = pd.read_csv(DATA_PATH_OUTPUT / '60models_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders


fullsequential_60 = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 60 Autoencoders
seq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_sequential.csv', index_col=0)
fullseq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_fullsequential.csv', index_col=0)
po_spe = pd.read_csv(DATA_PATH_OUTPUT / 'pollen_speech_200models_ensemble_number_of_auto.csv', index_col=0)
ca_ga_independent = pd.read_csv(DATA_PATH_OUTPUT / 'cardiogasdrift_200models_ensemble_number_of_auto.csv', index_col=0)
sat_ind = pd.read_csv(DATA_PATH_OUTPUT / 'satellite_200models_ensemble_number_of_auto.csv', index_col=0)
mapephowa_ind = pd.read_csv(DATA_PATH_OUTPUT / 'mapephowa_200models_ensemble_number_of_auto.csv', index_col=0)

syn = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'synchronizing_ver2.csv', index_col=0)
pho_pol_syn = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'phoneme_pollen_30models_synchronizing.csv', index_col=0)

def plotting_datasets(n):
    # Averaging AUC scores of all datasets for each autoencoder model in ensemble
    ind_60_mean = np.array(round(independent_ens_60.iloc[:, [n]], 4))
    seq_60_mean = np.array(round(sequential_60.iloc[:, [n]], 4))
    fullseq_60_mean = np.array(round(fullsequential_60.iloc[:, [n]], 4))

    # plotting
    # summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
    summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
    summary_60models['Number of Base Models'] = [i for i in range(1, 61)]
    summary_60models['Sequential Autoencoder Ensemble'] = seq_60_mean
    summary_60models['Full Sequential Autoencoder Ensemble'] = fullseq_60_mean
    summary_60models['Independent Autoencoder Ensemble'] = ind_60_mean

    #palette = sns.color_palette("mako_r", 3)
    #sns.lineplot(data = summary_60models[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble']])
    sns.lineplot(data = summary_60models[['Sequential Autoencoder Ensemble']])
    #plt.plot(ind_axis_x, ind_60_mean)
    #plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
    plt.legend(loc='best')
    plt.title(f'{LEGEND[n-1]}')
    plt.xlabel('Number of autoencoder models')
    plt.ylabel('AUC scores')
    plt.savefig(DATA_PATH_OUTPUT / 'datasets' / f'{LEGEND[n-1]}_summary.jpg')
    plt.show()

#plotting_datasets(3)

def plotting(n):
    # Averaging AUC scores of all datasets for each autoencoder model in ensemble
    seq_60_mean = np.array(round(seq_200models.iloc[:, [n]], 4))
    fullseq_60_mean = np.array(round(fullseq_200models.iloc[:, [n]], 4))
    #po_spe_mean = np.array(round(po_spe.iloc[:, [2]], 4))
    #ca_ga_mean = np.array(round(ca_ga_independent.iloc[:, [1]], 4))
    #sat_mean = np.array(round(sat_ind.iloc[:, [1]], 4))
    #mapephowa_mean = np.array(round(mapephowa_ind.iloc[:, [4]], 4))
    syn_mean = np.array(round(syn.iloc[:, [n]], 4))
    pho_pol_syn_mean = np.array(round(pho_pol_syn.iloc[:, [2]], 4))


    # plotting
    # summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
    #summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
    #summary_60models['Number of Base Models'] = [i for i in range(1, 201)]
    #summary_60models['Sequential Autoencoder Ensemble'] = seq_60_mean
    #summary_60models['Full Sequential Autoencoder Ensemble'] = fullseq_60_mean
    #summary_60models['Independent Autoencoder Ensemble'] = mapephowa_mean

    summary_syn = pd.DataFrame(columns=['Number of Base Models', 'Synchronizing Sequential Autoencoder Ensemble'])
    summary_syn['Number of Base Models'] = [i for i in range(1, 31)]
    summary_syn['Synchronizing Sequential Autoencoder Ensemble'] = pho_pol_syn_mean
    #sns.lineplot(data = summary_syn[['Synchronizing Sequential Autoencoder Ensemble']])


    #palette = sns.color_palette("mako_r", 3)
    #sns.lineplot(data = summary_60models[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble']])
    plt.plot([i for i in range(1, 31)], pho_pol_syn_mean, label='Synchronizing Sequential Autoencoder Ensemble')
    #plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
    plt.legend(loc='best')
    plt.title(f'{LEGEND[n-1]}')
    plt.xlabel('Number of autoencoder models')
    plt.ylabel('AUC scores')
    #plt.xlim(0, 5)
    plt.savefig(DATA_PATH_OUTPUT / 'synchronizing' / f'{LEGEND[n-1]}_syn_30models_plotting.jpg')
    plt.show()

plotting(7)


#ind_ensembl_200models = pd.DataFrame(columns=['number of Autoencoders','cardio', 'gas_drift', 'satellite', 'magic_telescope', 'pendigits', 'phoneme', 'pollen', 'speech', 'waveform'])
#ind_ensembl_200models['number of Autoencoders'] = [i for i in range(1, 201)]
#ind_ensembl_200models['cardio'] = ca_ga_independent.iloc[:, [1]]
#ind_ensembl_200models['gas_drift'] = ca_ga_independent.iloc[:, [2]]
#ind_ensembl_200models['satellite'] = sat_ind.iloc[:, [1]]
#ind_ensembl_200models['magic_telescope'] = mapephowa_ind.iloc[:, [1]]
#ind_ensembl_200models['pendigits'] = mapephowa_ind.iloc[:, [2]]
#ind_ensembl_200models['phoneme'] = mapephowa_ind.iloc[:, [3]]
#ind_ensembl_200models['pollen'] = po_spe.iloc[:, [1]]
#ind_ensembl_200models['speech'] = po_spe.iloc[:, [2]]
#ind_ensembl_200models['waveform'] = mapephowa_ind.iloc[:, [4]]
#
#ind_ensembl_200models_df = pd.DataFrame.from_dict(ind_ensembl_200models)
#ind_ensembl_200models_df.to_csv(DATA_PATH_OUTPUT / '200models_independent_ensemble.csv')
#

#seq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_sequential.csv', index_col=0)
#fullseq_200models = pd.read_csv(DATA_PATH_OUTPUT / 'rand42_200models_fullsequential.csv', index_col=0)
#ind_ens_200models = pd.read_csv(DATA_PATH_OUTPUT / '200models_independent_ensemble.csv', index_col=0)
#seq_200models_mean = np.array(round(seq_200models.iloc[:101, 1:].mean(axis=1), 4))
#fullseq_200models_mean = np.array(round(fullseq_200models.iloc[:101, 1:].mean(axis=1), 4))
#ind_ens_200models_mean = np.array(round(ind_ens_200models.iloc[:101, 1:].mean(axis=1), 4))
#
## plotting 200 models
## summary_60models = pd.DataFrame(columns=['Number of Base Models', 'Independent Ensemble', 'Sequential Ensemble', 'Full Sequential Ensemble'])
#summary_200models = pd.DataFrame(columns=['Number of Base Models', 'Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble'])
#summary_200models['Number of Base Models'] = [i for i in range(1, 102)]
#summary_200models['Sequential Autoencoder Ensemble'] = seq_200models_mean
#summary_200models['Full Sequential Autoencoder Ensemble'] = fullseq_200models_mean
#summary_200models['Independent Autoencoder Ensemble'] = ind_ens_200models_mean
#
# 
#
#sns.lineplot(data = summary_200models[['Sequential Autoencoder Ensemble', 'Full Sequential Autoencoder Ensemble', 'Independent Autoencoder Ensemble']])
##plt.plot(ind_axis_x, ind_60_mean)
##plt.scatter(ind_axis_x, ind_60_mean, alpha=0.5, label='Independent Ensemble')
#plt.legend(loc='best')
#plt.title('Correlation between number of autoencoders and mean of AUC scores in an ensemble model', fontsize=8)
#plt.xlabel('Number of autoencoder models')
#plt.ylabel('Mean of AUC scores')
#plt.savefig(DATA_PATH_OUTPUT / 'plott200' / 'fullsummary_100models.jpg')
#plt.show()