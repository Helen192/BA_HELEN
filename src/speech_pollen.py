# try to train model for speech and pollen dataset separately


# Training an ensemble of 10 different sequential models, in which each sequential model contains 10 different Autoencoders
# These sequential models are differentiated by random seed
# Then calculate the mean and standard deviation AUC scores of 10 ensemble models for each dataset
# AUC score is calculated by using the average value of all error predictions from all trained models

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

test_pollen = np.load(DATA_PATH_INPUT / 'pollen.npz')
test_speech = np.load(DATA_PATH_INPUT / 'speech.npz')


LATENT_LIST = [16, 256]


def speech_pollen_sequential(model_architecture, num_models, latent_list: list, data:list, legend:list, seed_list, epochs):
  """Create a syschronizing ensemble, in which the prediction errors of the last 
  trained model will be added as an feature to the original training data used for the next model.
  AUC score is calculated by using the error predictions of the last sychronized model
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    num_models -- the number of models which are integrated in one sychronizing ensemble model
    latent_list -- this contains the best size of latent space for each dataset in data respectively
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    seed_list
    epochs -- 
  Returns:
    drop_sequential_dict -- this contains the auc score list of each dataset in data according to each seed

  """
  drop_sequential_dict = {'dataset': legend, 'seed': seed_list, 'auc': []}
  for seed in seed_list:
    # Set random seed
    tf.keras.utils.set_random_seed(seed)
    dropsyn = []

    for n in range(len(data)):
      df = data[n] # extract dataset n in data
      # extract X_train, X_test, y_test in df and scaled X_train, X_test
      X_train, X_test, y_test = extract_data(df)

      # list of pred_errors. Each element contains all the difference between inputs and their predictions of one model
      avg_pred_errors_list = []
      test_errors_list = []
      auc_score_list = []
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
        autoencoder, history = training_model(model_architecture, X_plus_one, X_test_one, latent = latent_list[n], verbose = 0, loss = "mse", epochs=epochs, batch_size=128)
        
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
        
      dropsyn.append(auc_list)
    drop_sequential_dict['auc'].append(dropsyn)

    # extract AUC scores of each dataset from drop_sequential_dict 
  pollen_auc = [drop_sequential_dict['auc'][i][0] for i in range(0, 10)]
  speech_auc = [drop_sequential_dict['auc'][i][1] for i in range(0, 10)]
 

  # calculate mean of AUC score list
  pollen_mean = np.mean(np.array(pollen_auc), axis=0)
  speech_mean = np.mean(np.array(speech_auc), axis=0)

  # calculate standard deviation of AUC score list of each dataset
  pollen_std = np.std(np.array(pollen_auc), axis=0)
  speech_std = np.std(np.array(speech_auc), axis=0)

  drop_se = {'pollen mean': pollen_mean,
             'pollen std': pollen_std,
             'speech mean': speech_mean,
             'speech std': speech_std}
  
  # plot AUC score mean of each dataset according to the number of models in the sequential 
  x = [i for i in range(1, 11)]
  plt.plot(x, pollen_mean, label='pollen')
  plt.plot(x, speech_mean, label='speech')
  plt.title('Correlation between AUC score and the number of autoencoder model used in the sequential model')
  plt.xlabel('The number of autoencoder models used in the sequential model')
  plt.ylabel('AUC score')
  plt.legend(loc='best')
  plt.savefig(DATA_PATH_OUTPUT / 'speech_pollen_ensemble_ofsequential.pdf')
  plt.show()
  
  # this dataframe is the result of training 10 different full_sequential models, which are a combination of 10 autoencoder models 
  drop_sequential_df = pd.DataFrame.from_dict(drop_se)
  drop_sequential_df.to_csv(DATA_PATH_OUTPUT / 'speech_pollen_Ensemble_OfSequential.csv')  
  return drop_sequential_df


SEED_LIST = [1, 3, 42, 7, 10, 100, 123, 9, 50, 142]
speech_pollen_df = speech_pollen_sequential(Autoencoder, 10, LATENT_LIST, [test_pollen, test_speech], LEGEND,seed_list = SEED_LIST, epochs =2000)
print(speech_pollen_df)