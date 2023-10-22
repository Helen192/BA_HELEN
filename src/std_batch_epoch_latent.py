import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from autoencoder import *

# get AUC data
batch_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_batch_auc.csv', index_col=0)
epoch_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_epoch_auc.csv', index_col=0)
latent_auc = pd.read_csv(DATA_PATH_OUTPUT / 'inspect_latent_auc.csv', index_col=0)

batch_std = batch_auc.iloc[:, 1:].std().to_frame()
epoch_std = epoch_auc.iloc[:, 1:].std().to_frame()
latent_std = latent_auc.iloc[:,1:].std().to_frame()

batch = np.array(batch_std)[:,0]
epoch = np.array(epoch_std)
latent = np.array(latent_std)




#batch = [0.019025, 0.002465, 0.006580, 0.004559, 0.002574, 0.038842, 0.009556, 0.032427, 0.016662]
#epoch = [0.011723, 0.003547, 0.016751, 0.001550, 0.003996, 0.047053, 0.008961, 0.013796, 0.010299]
#latent = [0.011895, 0.029220, 0.023913, 0.012204, 0.004821, 0.032524, 0.011702, 0.016901, 0.011599]

batch_count = 4
epoch_count = 2
latent_count = 3


std_df = pd.DataFrame(columns=['dataset', 'batch_std', 'epoch_std', 'latent_std'])
std_df['dataset'] = LEGEND
std_df['batch_std'] = batch
std_df['epoch_std'] = epoch
std_df['latent_std'] = latent

print(std_df)
sns.set_palette('pastel')
#sns.barplot(std_df, estimator = np.median)
#plt.savefig(DATA_PATH_OUTPUT / 'std_batch_epoch_latent.jpg')

sns.barplot(std_df, ci=None)
plt.title('Hyperparameter influence for autoencoder model')
plt.savefig(DATA_PATH_OUTPUT / 'mean_of_std.jpg')
plt.show()
  
