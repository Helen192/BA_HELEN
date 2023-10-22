import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autoencoder import *

# get scores of all experiments related to synchronizing ensembles
to5_synchronizing = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'old' / 'synchronizing.csv', index_col=0) # the results of synchronizing ensemble contains till 5 Autoencoders
to10_sate_phone_poll = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'old' / 'test_synchronizing.csv', index_col=0) # the results of satellite, phoneme, pollen with synchronizing of 10 autoencoder
to15_phone_poll = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'old' / 'test_2000epoch_synchronizing.csv', index_col=0) # the results of phoneme, pollen with synchronizing of 15 autoencoders
to30_phone_poll = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'old' / 'test_30models_2000epoch_synchronizing.csv', index_col=0) # the results of phoneme, pollen with synchronizing of 30 autoencoders
ver2_synchronizing = pd.read_csv(DATA_PATH_OUTPUT / 'synchronizing' /'synchronizing_ver2.csv', index_col=0)

# get max score of each dataset trained by different synchronizing ensembles
to5_synchronizing_max = np.array(round(to5_synchronizing.iloc[:, 1:].max(axis=0), 4))
to10_sate_phone_poll_max = np.array(round(to10_sate_phone_poll.iloc[:, 1:].max(axis=0), 4))
to15_phone_poll_max = np.array(round(to15_phone_poll.iloc[:, 1:].max(axis=0), 4))
to30_phone_poll_max = np.array(round(to30_phone_poll.iloc[:, 1:].max(axis=0), 4))
ver2_synchronizing_max = np.array(round(ver2_synchronizing.iloc[:, 1:].max(axis=0), 4))

to10_max = [0, 0, 0.8501, 0, 0, 0.7476, 0.4907, 0, 0]
to15_max = [0, 0, 0, 0, 0, 0.7116, 0.4848, 0, 0]
to30_max = [0, 0, 0, 0, 0, 0.7377, 0.4864, 0, 0]

synchronizing_df = pd.DataFrame(columns=['dataset', 'synchronizing_5', 'synchronizing_10','synchronizing_15', 'synchronizing_30', 'ver2_syn'])
synchronizing_df['dataset'] = LEGEND
synchronizing_df['synchronizing_5'] = to5_synchronizing_max
synchronizing_df['synchronizing_10'] = to10_max
synchronizing_df['synchronizing_15'] = to15_max
synchronizing_df['synchronizing_30'] = to30_max
synchronizing_df['ver2_syn'] = ver2_synchronizing_max

# save result to .csv and export
#synchronizing_df.to_csv(DATA_PATH_OUTPUT / 'summary_synchronizing_results.csv')
print(synchronizing_df)

#print the best AUC score of each dataset trained by synchronizing
print(synchronizing_df.iloc[:, 1:].max(axis=1))
# print the average 
# the best AUC score of each dataset trained by synchronizing models
synchronizing_df_max = synchronizing_df.iloc[:, 1:].max(axis=1)
# the average of best AUC score of all datasets
synchronizing_mean = round(np.mean(synchronizing_df_max), 4)
ver2_mean = round(np.mean(ver2_synchronizing_max), 4)
print(f"Mean of max: {synchronizing_mean} vs. mean of Ver2: {ver2_mean}")
