# Full_Sequential uses predictions errors of all previous trained models as features added to the original training dataset to train the next model
# AUC score is calculated by using the average value of all prediction errors of trained models

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


def full_sequential(model_architecture, num_models, latent_list: list, data:list, legend:list, epochs):
  """Create a sequential syschronizing ensemble, in which the error of the all previous models will be used as features in the data used to train the next model.
  AUC score is calculated by using the average value of all error predictions from all previous sychronized model
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    num_models -- the number of models which are integrated in one sychronizing ensemble model
    latent_list -- this contains the best size of latent space for each dataset in data respectively
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    epochs -- 
  Returns:
    sesyn_dict -- this contains the auc score of each dataset in data
  """
  # Set random seed
  tf.keras.utils.set_random_seed(11111)
  sesyn_dict = {'dataset': legend, 'auc': []}

  for n in range(len(data)):
    df = data[n] # extract dataset n in data
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    
    # list of pred_errors. Each element contains all the difference between training data and their predictions of one model
    avg_pred_errors_list = []
    # list of test_errors. Each element contains all the difference between test data and their predictions on one model
    test_errors_list = []
    X = X_train.copy()
    X_te = X_test.copy()
    auc_list = []  # this list saves the AUC score of each model during training process

    for i in range(num_models):
      # 1. Create model
      # if pred_errors_list is not empty, then we have to concatenate the element at position i-1 to the current training input
      if avg_pred_errors_list:
        # convert the element at position i-1 in pred_errors_list to a 2D-tensor and then convert it to a numpy array. 
        pred_current = tf.reshape(avg_pred_errors_list[i-1], [-1, 1]).numpy()
        # convert the element at position i-1 in test_errors_list to a 2D-tensor and then convert it to a numpy array.
        pred_test_current = tf.reshape(test_errors_list[i-1], [-1, 1]).numpy()
        # concatenate the pred_current to X
        X = np.concatenate((X, pred_current), axis=1)
        # concatenate the pred_current to X_test also. This new X_test will be used for later predict method   
        X_te = np.concatenate((X_te, pred_test_current), axis=1)

      # training model
      autoencoder, history = training_model(model_architecture, X, X_te, latent = latent_list[n], verbose = 0, loss = "mse", epochs=epochs, batch_size=64)

      # getting predictions (recontructions) of the training data
      preds = autoencoder.predict(X)
      # calculate the difference between preds and the training data using (input - preds)**2
      pred_errors = tf.math.square(tf.math.subtract(preds, X))
      # calculate the average of pred_errors. We set axis = 1, so that the average is calculated by column-weise
      avg_preds = tf.math.reduce_mean(pred_errors, axis=1)   # shape = (X_train.shape[1], 1)
      # append avg_preds to the avg_pred_errors_list
      avg_pred_errors_list.append(avg_preds)

      # getting predictions (recontructions) of the test data
      preds_test = autoencoder.predict(X_te)
      # calculate the difference between preds and the test data using (input - preds)**2. These values is used to create a new feature added to the next training loop
      pred_errors_test = tf.math.square(tf.math.subtract(preds_test, X_te))
      # calculate the average of pred_errors_test. We set axis = 1, so that the average is calculated by column-weise
      avg_preds_test = tf.math.reduce_mean(pred_errors_test, axis=1)   # shape = (X_train.shape[1], 1)
      # append avg_preds_test to the test_errors_list
      test_errors_list.append(avg_preds_test)
      # calculate the average of all elements in test_errors_list up to index i. This average value is used to calculate AUC Scores
      tensor_errors_list  = tf.reshape(tf.math.reduce_mean(tf.convert_to_tensor(test_errors_list), axis=0), [y_test.shape[0], 1])

      # 4. Calculating AUC score of each training model 
      #avg_test_errors_list = 
      auc_score = roc_auc_score(y_test, tensor_errors_list)
      auc_list.append(auc_score)
      
    sesyn_dict['auc'].append(auc_list)
   # create a dictionary contains all AUC scores of all datasets corresponding to the number of autoencoders in the ensemble
  sesyn_auc = {'number of Autoencoders': list(range(1, num_models + 1)),
                'cardio': sesyn_dict['auc'][0],
                'gas-drift': sesyn_dict['auc'][1],
                'satellite': sesyn_dict['auc'][2],
                'magic_telescope': sesyn_dict['auc'][3],
                'pendigits': sesyn_dict['auc'][4],
                'phoneme': sesyn_dict['auc'][5],
                'pollen': sesyn_dict['auc'][6],
                'speech': sesyn_dict['auc'][7],
                'waveform': sesyn_dict['auc'][8]}

  # save result to .csv and export it
  sesyn_df = pd.DataFrame.from_dict(sesyn_auc)
  sesyn_df.to_csv(DATA_PATH_OUTPUT / 'rand11111_60models_fullsequential.csv')
  return sesyn_df


full_sequential_df = full_sequential(Autoencoder, num_models = 60, latent_list= LATENT_LIST, data = DATA, legend = LEGEND, epochs = 2000)
print(full_sequential_df)