# inspect the effect of ensemble model on datasets
    # We inspect the correlation between AUC score and the ensembel model, which contains models with the same architecture, 
    # but different random seed

# we train with epochs=2000, batch_size=64, best latent_size for each dataset: LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112]

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


LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112]

# for this function, seed_list is set up manually

def inspect_ensemble(model_architecture, seed_list, latent_list: list, data:list, legend:list, epochs):
  """
  This function show us AUC score of datasets trained by using ensemble model. Ensemble model contains models with the same architecture, but different random seed
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    seed_list -- a list of numbers which are used to set random seed for the model
    latent_list -- this contains the best size of latent space for each dataset in data respectively
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    epochs -- 
  Returns:
    plot
    ensemble_dict - this dictionary contains auc scores of datasets trained by ensemble model
  """
  ensemble_dict = {'dataset': legend, 'random_seed': seed_list, 'auc': []}

  for n in range(len(data)):
    df = data[n] # extract dataset n in data
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # create and train ensemble model for each dataset
    # list of pred_errors. Each element contains all the difference between inputs and their predictions of one model
    pred_errors_list = []

    for SEED in seed_list:
      # set random seed
      tf.keras.utils.set_random_seed(SEED)
      # create autoencoder model
      autoencoder, history = training_model(model_architecture, X_train, X_test, latent = latent_list[n], epochs=epochs)
      # getting predictions (recontructions) of the test data
      preds = autoencoder.predict(X_test)
      # calculate the difference between preds and test data using mean square error
      pred_errors = tf.keras.losses.mse(preds, X_test)
      # append pred_error to the pred_errors_list
      pred_errors_list.append(pred_errors)

    # convert pred_errors_list to a tensor
    preds_tensor = tf.convert_to_tensor(pred_errors_list)
    # calculate the average of all pred_errors in the pred_errors_list. We set axis=0, so that the average is calculated by row-weise
    avg_preds = tf.math.reduce_mean(preds_tensor, axis=0)
    auc_score = roc_auc_score(y_test, avg_preds)
    ensemble_dict['auc'].append(auc_score)

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, avg_preds)
    plt.plot(fpr, tpr, label = f"{legend[n]}: {auc_score}")
    
  plt.legend(loc='best')
  plt.title('ROC of ensemble models')
  plt.plot([0, 1], [0, 1],'r--')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  # save the plot
  plt.savefig(DATA_PATH_OUTPUT / 'ensembles.png')
  plt.show()

  # create a dictionary contains all AUC scores of all datasets for each epoch size value was used to train
  ensemble_auc = {'random_seed': seed_list,
                'cardio': ensemble_dict['auc'][0],
                'gas-drift': ensemble_dict['auc'][1],
                'satellite': ensemble_dict['auc'][2],
                'magic_telescope': ensemble_dict['auc'][3],
                'pendigits': ensemble_dict['auc'][4],
                'phoneme': ensemble_dict['auc'][5],
                'pollen': ensemble_dict['auc'][6],
                'speech': ensemble_dict['auc'][7],
                'waveform': ensemble_dict['auc'][8]}

  # save result to .csv and export it
  ensemble_df = pd.DataFrame.from_dict(ensemble_auc)
  ensemble_df.to_csv(DATA_PATH_OUTPUT / 'inspect_ensemble_auc.csv')
  return ensemble_dict


SEED_LIST = [0, 7, 42, 24, 100, 1234]
ensemble_dict = inspect_ensemble(Autoencoder, seed_list= SEED_LIST, latent_list= LATENT_LIST , data = DATA, legend = LEGEND, epochs = 2000)
print(ensemble_dict)