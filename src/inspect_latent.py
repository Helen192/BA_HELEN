# Inspect the effect of latent space on the models
    # We inspect the correlation between size of latent space and AUC score of models of different datasets
    # We set epochs=1500 and batch_size=128

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


def inspect_latent(model_architecture, latent_list: list, data:list, legend:list, epochs, batch):
  """
  This function show us the correlation between latent space and AUC score
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    epochs
    batch
  Returns:
    plot
    latent_dict -- this dictionary contains AUC scores of each datasets at each latent space
  """
  # Set random seed
  tf.keras.utils.set_random_seed(42)
  latent_dict = {'dataset': legend, 'auc': [], 'latent space': latent_list}

  for n in range(len(data)):
    df = data[n]
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # list contains auc_score of trained models
    auc_list = []
    # training model with different sizes of latent space
    for i in latent_list:
      model, hist = training_model(model_architecture, X_train, X_test, latent = i, loss = 'mse', epochs=epochs, batch_size=batch)

      #getting predictions (recontructions) of the test data
      preds = model.predict(X_test)
      # calculate the difference between predictions and test data using mean square error
      pred_errors = tf.keras.losses.mse(preds, X_test)
      # Check the prediction performance
      auc_score = roc_auc_score(y_test, pred_errors)
      auc_list.append(auc_score)

    latent_dict['auc'].append(auc_list)
    # plot the effect of latent space on trained models
    plt.plot(latent_list, auc_list)
    
  plt.legend(legend, loc='best')
  plt.title('Correlation between latent space and AUC scores of trained models')
  plt.xlabel("Latent size")
  plt.ylabel("AUC score")
  # save plot
  plt.savefig(DATA_PATH_OUTPUT / 'inspect_latent.jpg')
  plt.show()
  # create a dictionary contains all AUC scores of all datasets for each latent size value was used to train
  latent_auc = {'latent space': latent_list,
                'cardio': latent_dict['auc'][0],
                'gas-drift': latent_dict['auc'][1],
                'satellite': latent_dict['auc'][2],
                'magic_telescope': latent_dict['auc'][3],
                'pendigits': latent_dict['auc'][4],
                'phoneme': latent_dict['auc'][5],
                'pollen': latent_dict['auc'][6],
                'speech': latent_dict['auc'][7],
                'waveform': latent_dict['auc'][8]}

  # save result to .csv and export it
  latent_df = pd.DataFrame.from_dict(latent_auc)
  latent_df.to_csv(DATA_PATH_OUTPUT / 'inspect_latent_auc.csv')
  return latent_df

latent_df= inspect_latent(Autoencoder, latent_list = [4, 8, 16, 32, 64, 96, 112, 128, 144, 160, 192, 256, 320] , data = DATA, legend = LEGEND, epochs=2000, batch=64)
print(latent_df)

