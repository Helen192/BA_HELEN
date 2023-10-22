# Correlation between latent space and epochs
    # we try to figure out the relationship between latent space and epochs in a model.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from autoencoder import *

def heatmap_epoch_latent(model_architecture, epoch_list: list, latent_list: list, data:list, legend:list, batch):
    """
    This function show us the influence of both latent space and epochs on AUS scores of trained models. 
    we can see the correlation between latent and epochs through the changes of AUC scores. The correlation is shown on the heatmap
    
    Arguments:
      model_architecture -- the model architecture that we want to use to create model
      data -- a list of datasets
      latent_list  - this contains the latent space of each dataset respectively
      legend -- name of datasets used to display on plot
      epoch_list -- a list of different epochs
    Returns:
      plot -- heatmap for each dataset
      lat_epo -- this dictionary contains AUC score of each dataset according to each epoch value and latent space
    """
    # Set random seed
    tf.keras.utils.set_random_seed(42)
    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(3, 3, figsize=(25, 15), tight_layout=True)
    ax = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    for n in range(len(data)):
        lat_epo= {'epoch': epoch_list, 'latent': latent_list, 'auc': []}
        df = data[n]
        # extract X_train, X_test, y_test in df and scaled X_train, X_test
        X_train, X_test, y_test = extract_data(df)
    
        # training model with different sizes of latent space
        for i in range(len(epoch_list)):
            # list contains auc_score of trained models
            auc_list = []
            for j in range(len(latent_list)):
                model, hist = training_model(model_architecture, X_train, X_test, latent = latent_list[j], loss = 'mse', epochs=epoch_list[i], batch_size=batch)

                #getting predictions (recontructions) of the test data
                preds = model.predict(X_test)
                # calculate the difference between predictions and test data using mean square error
                pred_errors = tf.keras.losses.mse(preds, X_test)
                # Check the prediction performance
                auc_score = roc_auc_score(y_test, pred_errors)
                auc_list.append(auc_score)

            lat_epo['auc'].append(auc_list)

        # create a dataframe
        df = pd.DataFrame(lat_epo['auc'])
        # transpose of dataframe
        df = df.transpose()
        # change the column names
        df.columns = epoch_list
        # change the row indexes
        df.index = latent_list
        df.to_csv(DATA_PATH_OUTPUT / f'{legend[n]}_heatmap_epoch_latent.csv')
        # plot heatmap
        sns.heatmap(df,annot=True, linecolor='White', cmap='RdPu', linewidth=1.5, cbar=False, ax=ax[n])
        ax[n].set(title=f'{legend[n]}', xlabel='Epochs', ylabel='Latent')

    plt.savefig(DATA_PATH_OUTPUT / f'heatmap_latent_epoch.jpg')
    plt.savefig(DATA_PATH_OUTPUT / f'heatmap_latent_epoch.pdf')
    plt.plot

epochs = [50, 200, 500, 1000, 1500, 2000, 2500, 3000]
latents = [4, 16, 32, 64, 96, 112, 160, 256]
heatmap_epoch_latent(Autoencoder, epoch_list=epochs, latent_list=latents, data = DATA, legend=LEGEND, batch=64)

