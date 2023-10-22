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


def latent_epochs(model_architecture, epoch_list: list, latent_list: list, data:list, legend:list, batch):
  """
  This function show us the influence of both latent space and epochs on AUS scores of trained models. Also we can see the correlation between latent and epochs through the changes of AUC scores 
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    data -- a list of datasets
    latent_list  - this contains the latent space of each dataset respectively
    legend -- name of datasets used to display on plot
    epoch_list -- a list of different epochs
  Returns:
    plot
    lat_epo -- this dictionary contains AUC score of each dataset according to each epoch value and latent space
  """
  # Set random seed
  tf.keras.utils.set_random_seed(42)
  lat_epo= {'dataset': legend, 'auc': [], 'latent': latent_list,'epochs': epoch_list}

  for n in range(len(data)):
    df = data[n]
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # list contains auc_score of trained models
    auc_list = []
    # training model with different sizes of latent space
    for i in range(len(epoch_list)):
      model, hist = training_model(model_architecture, X_train, X_test, latent = latent_list[i], loss = 'mse', epochs=epoch_list[i], batch_size=batch)

      #getting predictions (recontructions) of the test data
      preds = model.predict(X_test)
      # calculate the difference between predictions and test data using mean square error
      pred_errors = tf.keras.losses.mse(preds, X_test)
      # Check the prediction performance
      auc_score = roc_auc_score(y_test, pred_errors)
      auc_list.append(auc_score)

    lat_epo['auc'].append(auc_list)

  return lat_epo


EPOCHS = [50, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

lat_epo = latent_epochs(Autoencoder, epoch_list=EPOCHS, latent_list = [4, 8, 16, 32, 64, 96, 112, 128, 144, 160, 192, 256], data = DATA, legend = LEGEND, batch=64)

# create a dictionary contains all AUC scores of all datasets for each latent size value was used to train
latent_epo = {'latent_size': lat_epo['latent'],
              'epoch_size': lat_epo['epochs'],
               'cardio': lat_epo['auc'][0],
               'gas_drift': lat_epo['auc'][1],
               'satellite': lat_epo['auc'][2],
               'magic_telescope': lat_epo['auc'][3],
               'pendigits': lat_epo['auc'][4],
               'phoneme': lat_epo['auc'][5],
               'pollen': lat_epo['auc'][6],
               'speech': lat_epo['auc'][7],
               'waveform': lat_epo['auc'][8]}

latent_epo_df= pd.DataFrame.from_dict(latent_epo)
latent_epo_df_copy = latent_epo_df.copy()  
# save the result to .csv file
latent_epo_df_copy.to_csv(DATA_PATH_OUTPUT / 'correlation_latent_epoch.csv')


# plot the effect of latent space on trained models
fig, axis = plt.subplots()
axis.plot(latent_epo_df.iloc[:, 2], label='cardio')
axis.plot(latent_epo_df.iloc[:, 3], label='gas-drift')
axis.plot(latent_epo_df.iloc[:, 4], label='satellite')
axis.plot(latent_epo_df.iloc[:, 5], label='magic_telescope')
axis.plot(latent_epo_df.iloc[:, 6], label='pendigits')
axis.plot(latent_epo_df.iloc[:, 7], label='phoneme')
axis.plot(latent_epo_df.iloc[:, 8], label='pollen')
axis.plot(latent_epo_df.iloc[:, 9], label='speech')
axis.plot(latent_epo_df.iloc[:, 10], label='waveform')
plt.xticks(ticks= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],labels=EPOCHS)
plt.legend(loc='best')
plt.title('Dependency of AUC scores of trained models on both latent and epoch sizes')
plt.xlabel("epoch size")
plt.ylabel('AUC score')
axis2 = axis.twiny()
plt.xlabel("latent size")
plt.xticks(ticks= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], labels=[4, 8, 16, 32, 64, 96, 112, 128, 144, 160, 192, 256])
plt.savefig(DATA_PATH_OUTPUT / 'correlation_latent_epoch.jpg')
plt.show()
