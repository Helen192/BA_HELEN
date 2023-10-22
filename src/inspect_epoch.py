# Correlation between Autoencoder models and the number of epochs
# We try to figure out the relationship between AUC score of Autoencoder models with epochs

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

# old used latents
#LATENT_LIST = [4, 112, 64, 8, 8, 4, 4, 128, 64]

LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112]


def inspect_epochs(model_architecture, epoch_list: list, latent_list: list, data:list, legend:list):
  """
  This function show us the correlation between epochs and AUC score
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    data -- a list of datasets
    latent_list  - this contains the latent space of each dataset respectively
    legend -- name of datasets used to display on plot
    epoch_list -- a list of different epochs
  Returns:
    plot
    epoch_dict -- this dictionary contains AUC score of each dataset according to each epoch value
  """
  # Set random seed
  tf.keras.utils.set_random_seed(42)
  epoch_dict = {'dataset': legend, 'auc': [], 'epochs': epoch_list}

  for n in range(len(data)):
    df = data[n]
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # list contains auc_score of trained models
    auc_list = []
    # training model with different sizes of latent space
    for epoch in epoch_list:
      model, hist = training_model(model_architecture, X_train, X_test, latent = latent_list[n], loss = 'mse', epochs=epoch, batch_size=64)

      #getting predictions (recontructions) of the test data
      preds = model.predict(X_test)
      # calculate the difference between predictions and test data using mean square error
      pred_errors = tf.keras.losses.mse(preds, X_test)
      # Check the prediction performance
      auc_score = roc_auc_score(y_test, pred_errors)
      auc_list.append(auc_score)

    epoch_dict['auc'].append(auc_list)
    # plot the effect of latent space on trained models
    plt.plot(epoch_list, auc_list)
    
  plt.legend(legend, loc='best')
  plt.title('Correlation between epochs and AUC scores of trained models')
  plt.xlabel("epoch size")
  plt.ylabel("AUC score")
  # save plot
  plt.savefig(DATA_PATH_OUTPUT / 'inspect_epoch.jpg')
  plt.show()
  
  # create a dictionary contains all AUC scores of all datasets for each epoch size value was used to train
  epoch_auc = {'epoch_size': epoch_list,
                'cardio': epoch_dict['auc'][0],
                'gas-drift': epoch_dict['auc'][1],
                'satellite': epoch_dict['auc'][2],
                'magic_telescope': epoch_dict['auc'][3],
                'pendigits': epoch_dict['auc'][4],
                'phoneme': epoch_dict['auc'][5],
                'pollen': epoch_dict['auc'][6],
                'speech': epoch_dict['auc'][7],
                'waveform': epoch_dict['auc'][8]}

  # save result to .csv and export it
  epoch_df = pd.DataFrame.from_dict(epoch_auc)
  epoch_df.to_csv(DATA_PATH_OUTPUT / 'inspect_epoch_auc.csv')
  return epoch_df

epoch_df = inspect_epochs(Autoencoder, epoch_list=[50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], latent_list=LATENT_LIST, data = DATA, legend = LEGEND)
print(epoch_df)