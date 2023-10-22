import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

# get scores of all experiments related to Autoencoder models
auto_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_autoencoder_auc.csv', index_col=0)
batch_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_batch_auc.csv', index_col=0)
epoch_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_epoch_auc.csv', index_col=0)
latent_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_latent_auc.csv', index_col=0)
unoptimal_auto = pd.read_csv(DATA_PATH_OUTPUT / 'not_finetun_inspect_autoencoder_auc.csv', index_col=0) # model uses: LATENT_LIST = [4, 32, 16, 4, 4, 4, 4, 64, 32], epoch=1000
ver2_unoptimal_auto = pd.read_csv(DATA_PATH_OUTPUT / 'not_finetun_version2_inspect_autoencoder_auc.csv', index_col=0) #LATENT_LIST_Version2 = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoc=1000
randnet_auc = pd.read_csv(DATA_PATH_OUTPUT / 'Randnet_auc.csv', index_col=0)  # training by using randnet's architecture

unoptimal_300epochs = pd.read_csv(DATA_PATH_OUTPUT / '300epochs_100batch_not_finetun_version2_inspect_autoencoder_auc.csv', index_col=0)  # #LATENT_LIST_Version2 = [4, 16, 16, 4, 4, 4, 4, 256, 32], epoch=300, batch=100


batch_max = round(batch_auc.iloc[:, 1:].max(axis=0), 4)
epoch_max = round(epoch_auc.iloc[:, 1:].max(axis=0), 4)
latent_max = round(latent_auc.iloc[:, 1:].max(axis=0), 4)


batch = np.array(batch_max)
epoch = np.array(epoch_max)
latent = np.array(latent_max)

auto_df = pd.DataFrame(columns=['dataset', 'Randnet', 'unoptimal_300epochs','autoencoder_max', 'unoptimal_auto_max', 'version2_unoptimal_auto', 'batch_max','epoch_max', 'latent_max'])
auto_df['dataset'] = LEGEND
auto_df['Randnet'] = round(randnet_auc['AUC'],4)
auto_df['unoptimal_300epochs'] = round(unoptimal_300epochs.iloc[:, 2],4)
auto_df['autoencoder_max'] = round(auto_auc.iloc[:, 2],4)
auto_df['unoptimal_auto_max'] = round(unoptimal_auto.iloc[:, 2],4)
auto_df['version2_unoptimal_auto'] = round(ver2_unoptimal_auto.iloc[:, 2],4)
auto_df['batch_max'] = batch
auto_df['epoch_max'] = epoch
auto_df['latent_max'] = latent

# save result to .csv and export
#auto_df.to_csv(DATA_PATH_OUTPUT / 'summary_autoencoder_results.csv')
print(auto_df)

#print the best AUC score of each dataset trained by autoencoder model
print(round(auto_auc.iloc[:, 2],4))
# print the average 
auto_df_mean = round(np.mean(auto_auc.iloc[:, 2]), 4)
unoptimal_auto_df_mean = round(np.mean(unoptimal_auto.iloc[:, 2]), 4)
ver2_unoptimal_auto_df_mean = round(np.mean(ver2_unoptimal_auto.iloc[:, 2]), 4)
randnet_mean = round(np.mean(randnet_auc['AUC']), 4)
unoptimal_300epochs_mean = round(np.mean(unoptimal_300epochs['auc']), 4)
print(f"Randnet: {randnet_mean} vs 300epochs {unoptimal_300epochs_mean} vs {auto_df_mean} vs unoptimal {unoptimal_auto_df_mean} vs version2 unoptimal {ver2_unoptimal_auto_df_mean}")