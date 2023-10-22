import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *
from scipy.stats import wilcoxon

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

# extract data: sequential = [cardio L1, cardio L2, cardio L3, cardio L4, cardio L5, gas-drift L1, ...]
sequential_df = []
for n in range(1, 10):
    se_l1 = np.array(round(to60_sequential.iloc[:, [n]].max(axis=0), 4))
    sequential_df.append(se_l1[0])
    se_l2 = np.array(round(unoptimal_to60_sequential.iloc[:, [n]].max(axis=0), 4))
    sequential_df.append(se_l2[0])
    se_l3 = np.array(round(ver2_unoptimal_to60_sequential.iloc[:, [n]].max(axis=0), 4))
    sequential_df.append(se_l3[0])
    se_l4 = np.array(round(unoptimal_300epochs_sequential.iloc[:, [n]].max(axis=0), 4))
    sequential_df.append(se_l4[0])
    se_l5 = np.array(round(ver3_sequential.iloc[:, [n]].max(axis=0), 4))
    sequential_df.append(se_l5[0])



#3. Full Sequential
# get scores of all experiments related to full sequential ensembles
#to16_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'full_sequential.csv', index_col=0) # the results of sequential ensemble contains till 16 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / '60models_full_sequential.csv', index_col=0) # to 60 models, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
unoptimal_to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'ver1_unoptimal_60models_full_sequential.csv', index_col=0) # model uses: LATENT_LIST = [4, 32, 16, 4, 4, 4, 4, 64, 32], epoch=1000, epoch=1000, batch=64
ver2_unoptimal_to60_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'unoptimal_60models_full_sequential.csv', index_col=0) #  unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=1000, batch=64
unoptimal_300epochs_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batchunoptimal_60models_full_sequential.csv', index_col=0) # unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32], epochs=300, batch=100
ver3_fullsequential = pd.read_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_full_sequential_500e_64b.csv', index_col=0)

# extract data: fullsequential = [cardio L1, cardio L2, cardio L3, cardio L4, cardio L5, gas-drift L1, ...]
fullsequential_df = []
for n in range(1, 10):
    fu_l1 = np.array(round(to60_fullsequential.iloc[:, [n]].max(axis=0), 4))
    fullsequential_df.append(fu_l1[0])
    fu_l2 = np.array(round(unoptimal_to60_fullsequential.iloc[:, [n]].max(axis=0), 4))
    fullsequential_df.append(fu_l2[0])
    fu_l3 = np.array(round(ver2_unoptimal_to60_fullsequential.iloc[:, [n]].max(axis=0), 4))
    fullsequential_df.append(fu_l3[0])
    fu_l4 = np.array(round(unoptimal_300epochs_fullsequential.iloc[:, [n]].max(axis=0), 4))
    fullsequential_df.append(fu_l4[0])
    fu_l5 = np.array(round(ver3_fullsequential.iloc[:, [n]].max(axis=0), 4))
    fullsequential_df.append(fu_l5[0])



#4. Independent Ensemble
# get scores of all experiments related to independent ensemble model
inspect_ens = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_ensemble_auc.csv', index_col=0)  # the results of ensemble contains 6 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
ens_num = pd.read_csv(DATA_PATH_OUTPUT / 'ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 16 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
to60_ens = pd.read_csv(DATA_PATH_OUTPUT / 'optimal_to60 models_ensemble_number_of_auto.csv', index_col=0)  # the results of ensemble contains till 60 Autoencoders, optimal - LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112], epoch=2000, batch=64
ver1_ens = pd.read_csv(DATA_PATH_OUTPUT / 'ver1_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
ver2_ens = pd.read_csv(DATA_PATH_OUTPUT / 'ver2_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
unoptimal_ens = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batch_unoptimal_to60 models_ensemble_number_of_auto.csv', index_col=0)
ver3_ens = pd.read_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_ind_ensemble_500e_64b.csv', index_col=0)


# extract data: independent = [cardio L1, cardio L2, cardio L3, cardio L4, cardio L5, gas-drift L1, ...]
independent_df = []
for n in range(1, 10):
    # get max score of each dataset trained by different ensembles
    ind_l1 = np.array(round(to60_ens.iloc[-1:, [n]].max(axis=0), 4))
    independent_df.append(ind_l1[0])
    ind_l2 = np.array(round(ver1_ens.iloc[-1:, [n]].max(axis=0), 4))
    independent_df.append(ind_l2[0])
    ind_l3 = np.array(round(ver2_ens.iloc[-1:, [n]].max(axis=0), 4))
    independent_df.append(ind_l3[0])
    ind_l4 = np.array(round(unoptimal_ens.iloc[-1:, [n]].max(axis=0), 4))
    independent_df.append(ind_l4[0])
    ind_l5 = np.array(round(ver3_ens.iloc[-1:, [n]].max(axis=0), 4))
    independent_df.append(ind_l5[0])

wilcoxon_df = pd.DataFrame(columns=['Sequential', 'Full_Sequential', 'Independent'])
wilcoxon_df['Sequential'] = sequential_df
wilcoxon_df['Full_Sequential'] = fullsequential_df
wilcoxon_df['Independent'] = independent_df

#wilcoxon_df.to_csv(DATA_PATH_OUTPUT / 'wilcoxontest' / 'data_for_test.csv')

# 5. Test Wilcoxon : two-side: hypothese: Ho_ two groups are significantly similar  vs. H1_ two groups are not similar
# 5.1. Independent vs. Sequential
ind_seq = wilcoxon(independent_df, sequential_df)

# 5.2. Independent vs. Full Sequential
ind_full = wilcoxon(independent_df, fullsequential_df)

print(f"Test of ind vs seq: {ind_seq}")
print(f"Test of ind vs full: {ind_full}")

# Result:
# Test of ind vs seq: WilcoxonResult(statistic=107.0, pvalue=4.917420710626175e-07)  = 0.0000004917
# Test of ind vs full: WilcoxonResult(statistic=136.5, pvalue=4.335469782290602e-06) = 0.000004335

test = wilcoxon(sequential_df, fullsequential_df)
print(test)
