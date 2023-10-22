# Correlation between batch_size and AUC scores of models
# we train autoencoders with epochs = 1500 and different batch_size

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

def inspect_batchsize(model_architecture, batch_size: list, latent_list: list, data:list, legend:list, epoch):
  """
  This function show us the correlation between epochs and AUC score
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    batch_size -- a list contains different batch sizes
    data -- a list of datasets
    latent_list  - this contains the latent space of each dataset respectively
    legend -- name of datasets used to display on plot
  Returns:
    plot
    batch_dict -- this dictionary contains AUC score of each dataset according to each batch size
  """
  # Set random seed
  tf.keras.utils.set_random_seed(42)
  batch_dict = {'dataset': legend, 'auc': [], 'batch_size': batch_size}

  for n in range(len(data)):
    df = data[n]
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # list contains auc_score of trained models
    auc_list = []
    # training model with different batch sizes
    for batch in batch_size:
      model, hist = training_model(model_architecture, X_train, X_test, latent = latent_list[n], loss = 'mse', epochs=epoch, batch_size=batch)

      #getting predictions (recontructions) of the test data
      preds = model.predict(X_test)
      # calculate the difference between predictions and test data using mean square error
      pred_errors = tf.keras.losses.mse(preds, X_test)
      # Check the prediction performance
      auc_score = roc_auc_score(y_test, pred_errors)
      auc_list.append(auc_score)

    batch_dict['auc'].append(auc_list)
    # plot the effect of latent space on trained models
    plt.plot(batch_size, auc_list)
    
  plt.legend(legend, loc='best')
  plt.title('Correlation between batch size and AUC scores of trained models')
  plt.xlabel("batch size")
  plt.ylabel("AUC score")
  # save plot
  plt.savefig(DATA_PATH_OUTPUT / 'inspect_batchsize.jpg')
  plt.show()
  # create a dictionary contains all AUC scores of all datasets for each batch size value was used to train
  batch_auc = {'batch_size': batch_size,
                'cardio': batch_dict['auc'][0],
                'gas-drift': batch_dict['auc'][1],
                'satellite': batch_dict['auc'][2],
                'magic_telescope': batch_dict['auc'][3],
                'pendigits': batch_dict['auc'][4],
                'phoneme': batch_dict['auc'][5],
                'pollen': batch_dict['auc'][6],
                'speech': batch_dict['auc'][7],
                'waveform': batch_dict['auc'][8]}

  # save result to .csv and export it
  batch_df = pd.DataFrame.from_dict(batch_auc)
  batch_df.to_csv(DATA_PATH_OUTPUT / 'inspect_batch_auc.csv')
  return batch_df

batch_df= inspect_batchsize(Autoencoder, batch_size=[16, 32, 64, 96, 128, 256, 512], latent_list=LATENT_LIST, data = DATA, legend = LEGEND, epoch=2000)
print(batch_df)
