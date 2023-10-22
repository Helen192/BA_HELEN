# only the prediction errors of the last trained model are used as a feature in the data used to train the next model
# AUC score is calculated by using the average value of all prediction errors from all trained models

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


def sequential(model_architecture, num_models, latent_list: list, data:list, legend:list, epochs):
  """Create a sequential ensemble, in which the prediction errors of the last 
  trained model will be added as an feature to the original training data used for the next model.
  AUC score is calculated by using the prediction errors of all trained models
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    num_models -- the number of models which are integrated in one sychronizing ensemble model
    latent_list -- this contains the best size of latent space for each dataset in data respectively
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    epochs -- 
  Returns:
    dropsyn_dict -- this contains the auc score of each dataset in data
  """
  # Set random seed
  tf.keras.utils.set_random_seed(1234)
  dropsyn_dict = {'dataset': legend, 'auc': []}

  for n in range(len(data)):
    df = data[n] # extract dataset n in data
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)

    # list of pred_errors. Each element contains all the difference between inputs and their predictions of one model
    avg_pred_errors_list = []
    test_errors_list = []
    X_plus_one = X_train.copy()
    X_test_one = X_test.copy()
    auc_list = []  # this list saves the AUC score of each model during training process

    for i in range(num_models):
      # 1. Create model
      # if pred_errors_list is not empty, then we have to concatenate the element at position i to the current training input
      if avg_pred_errors_list:
        # convert the element at position i-1 in pred_errors_list to a 2D-tensor and then convert it to a numpy array. 
        pred_current = tf.reshape(avg_pred_errors_list[i-1], [-1, 1]).numpy()
        # only the prediction errors of the last model is concatenated to the original training data before training the next one
        X_plus_one = np.concatenate((X_train.copy(), pred_current), axis=1)
        # convert the element at position i-1 in test_errors_list to a 2D-tensor and then convert it to a numpy array. 
        pred_test_current = tf.reshape(test_errors_list[i-1], [-1, 1]).numpy()
        # concatenate the pred_current to X_test also. This new X_test will be used for later predict method   
        X_test_one = np.concatenate((X_test.copy(), pred_test_current), axis=1)
      
      # training model
      autoencoder, history = training_model(model_architecture, X_plus_one, X_test_one, latent = latent_list[n], verbose = 0, loss = "mse", epochs=epochs, batch_size=64)
      
      # getting predictions of the training data
      preds = autoencoder.predict(X_plus_one)
      # calculate the difference between preds and the training data using (input - preds)**2
      pred_errors = tf.math.square(tf.math.subtract(preds, X_plus_one))
      # calculate the average of pred_errors. We set axis = 1, so that the average is calculated by column-weise
      avg_preds = tf.math.reduce_mean(pred_errors, axis=1)   # shape = (X_train.shape[1], 1)
      # append avg_preds to the avg_pred_errors_list
      avg_pred_errors_list.append(avg_preds)

      # getting predictions (recontructions) of the test data
      preds_test = autoencoder.predict(X_test_one)
      # calculate the difference between preds and the test data using (input - preds)**2. These values is used to create a new feature added to the next training loop
      pred_errors_test = tf.math.square(tf.math.subtract(preds_test, X_test_one))
      # calculate the average of pred_errors_test. We set axis = 1, so that the average is calculated by column-weise
      avg_preds_test = tf.math.reduce_mean(pred_errors_test, axis=1)   # shape = (X_train.shape[1], 1)
      # append avg_preds_test to the test_errors_list
      test_errors_list.append(avg_preds_test)

      # calculate the average of all elements in test_errors_list up to index i. This average value is used to calculate AUC Scores
      tensor_errors_list  = tf.reshape(tf.math.reduce_mean(tf.convert_to_tensor(test_errors_list), axis=0), [-1, 1])
      # 4. Calculating AUC score of each training model 
      auc_score = roc_auc_score(y_test, tensor_errors_list)
      auc_list.append(auc_score)
      
    dropsyn_dict['auc'].append(auc_list)
   # create a dictionary contains all AUC scores of all datasets corresponding to the number of autoencoders in the ensemble
  dropsyn_auc = {'number of Autoencoders': list(range(1, num_models + 1)),
                'cardio': dropsyn_dict['auc'][0],
                'gas-drift': dropsyn_dict['auc'][1],
                'satellite': dropsyn_dict['auc'][2],
                'magic_telescope': dropsyn_dict['auc'][3],
                'pendigits': dropsyn_dict['auc'][4],
                'phoneme': dropsyn_dict['auc'][5],
                'pollen': dropsyn_dict['auc'][6],
                'speech': dropsyn_dict['auc'][7],
                'waveform': dropsyn_dict['auc'][8]}

  # save result to .csv and export it
  dropsyn_df = pd.DataFrame.from_dict(dropsyn_auc)
  dropsyn_df.to_csv(DATA_PATH_OUTPUT / 'rand1234_60models_sequential.csv')  
  return dropsyn_df

#dropsyn_df = sequential(Autoencoder, 60, latent_list= LATENT_LIST, data = DATA, legend = LEGEND, epochs = 2000)
#unoptimal_latent = [4, 16, 16, 4, 4, 4, 4, 256, 32]
#dropsyn_df = sequential(Autoencoder, 60, latent_list= unoptimal_latent, data = DATA, legend = LEGEND, epochs = 1000)
#dropsyn_df = sequential(Autoencoder, 60, latent_list= unoptimal_latent, data = DATA, legend = LEGEND, epochs = 300)

#ver1_unoptimal_latent = [4, 32, 16, 4, 4, 4, 4, 64, 32]
#dropsyn_df = sequential(Autoencoder, 60, latent_list= ver1_unoptimal_latent, data = DATA, legend = LEGEND, epochs = 1000)


dropsyn_df = sequential(Autoencoder, 60, latent_list= LATENT_LIST, data = DATA, legend = LEGEND, epochs = 2000)
print(dropsyn_df)