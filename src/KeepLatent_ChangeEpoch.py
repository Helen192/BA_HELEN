# For each dataset:
    # - we set their latent_size = alpha * number_features. This means latent_size of each dataset is a constant
    # - then we train models with different epoch_size and see how AUC score of that model changes


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


# By using this function, we can set the latent size for each dataset in a relative relationship with it's number of features

def keep_latent(model_architecture, epoch_list: list,  data:list, legend:list, alpha, batch):
  """
  This function inspects the correlation between latent and epochs through set a fix latent_size for each dataset and changes epoch_size.
  It aims to display how AUC scores are changed according to the epochs
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    data -- a list of datasets
    latent_list  - this contains the latent space of each dataset respectively
    legend -- name of datasets used to display on plot
    epoch_list -- a list of different epochs
    alpha -- the coefficient of number of features of a dataset. This parameter is used to calculate the latent size 
  Returns:
    plot
    epoch_dict -- this dictionary contains the AUC scores and epochs as well as latent size of each dataset
  """
  # get the number of feature of each dataset
  features = [df['x'].shape[1] for df in data]
  # calculate the latent list: laten_size = alpha * number of features
  latent_list = [int(i * alpha) for i in features ]

  # Set random seed
  tf.keras.utils.set_random_seed(42)

  epoch_dict = {'dataset': legend, 'auc': [], 'epochs': epoch_list, 'latent_size': latent_list}

  for n in range(len(data)):
    df = data[n]
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # list contains auc_score of trained models
    auc_list = []
    # training model with different sizes of latent space
    for epoch in epoch_list:
      model, hist = training_model(model_architecture, X_train, X_test, latent = latent_list[n], loss = 'mse', epochs=epoch, batch_size=batch)

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
  plt.title(f'Correlation between unchanged latent and different epochs (with alpha = {alpha})')
  plt.xlabel("epoch size")
  plt.ylabel("AUC score")
  # save plot
  plt.savefig(DATA_PATH_OUTPUT / 'keeplatent_changeepoch.jpg')
  plt.show()
  
  # create a dictionary contains all AUC scores of all datasets for each latent size value was used to train
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
  epoch_df.to_csv(DATA_PATH_OUTPUT / 'keeplatent_changeepoch.csv')
  return epoch_df


EPOCH_LIST = [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
keeplatent_changeepoch = keep_latent(Autoencoder, epoch_list = EPOCH_LIST, data = DATA, legend = LEGEND, alpha=0.5, batch=64)
print(keeplatent_changeepoch)
