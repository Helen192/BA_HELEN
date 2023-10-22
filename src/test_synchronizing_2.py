# train again some special datasets with sychronizing architecture

# All the prediction errors of the trained model are used as features in the data used to train the next model. 
# In this case, we keep the shape of prediction errors instead of reshape them to get only one feature
# For each dataset:
    # - we set their latent_size = alpha * number_features.

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

def test_synchronizing(model_architecture, num_models, data:list, legend:list, latent_list, epochs):
  """Create a syschronizing ensemble, in which the prediction errors of the previous 
  trained models will be added as an feature to the original training data used for the next model.
  AUC score is calculated by using the average value of all error predictions from all previous sychronized model
  
  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    num_models -- the number of models which are integrated in one sychronizing ensemble model
    latent_list -- this contains the best size of latent space for each dataset in data respectively
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    epochs -- 
    latent_list -- this contains the best size of latent space for each dataset in data respectively 
  Returns:
    auc_dict -- this contains the auc score of each dataset in data
  """

  # Set random seed
  tf.keras.utils.set_random_seed(42)
  fullsyn_dict = {'dataset': legend, 'auc': []}

  for n in range(len(data)):
    df = data[n] # extract dataset n in data
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)

    # list of pred_errors. Each element contains all the difference between inputs and their predictions of one model
    pred_errors_list = []
    test_errors_list = []
    errors_list = []
    X_plus_one = X_train.copy()
    X_test_one = X_test.copy()
    auc_list = []  # this list saves the AUC score of each model during training process


    for i in range(num_models):
      # 1. Create model
      # if pred_errors_list is not empty, then we have to concatenate the element at position i to the current training input
      if pred_errors_list:
        # convert the element at position i-1 in pred_errors_list to a 2D-tensor and then convert it to a numpy array. 
        pred_current = tf.reshape(pred_errors_list[i-1], [X_train.shape[0], -1]).numpy()
        # only the prediction errors of the last model is concatenated to the original training data before training the next one
        X_plus_one = np.concatenate((X_train.copy(), pred_current), axis=1)
        # convert the element at position i-1 in test_errors_list to a 2D-tensor and then convert it to a numpy array. 
        pred_test_current = tf.reshape(test_errors_list[i-1], [X_test.shape[0], -1]).numpy()
        # concatenate the pred_current to X_test also. This new X_test will be used for later predict method   
        X_test_one = np.concatenate((X_test.copy(), pred_test_current), axis=1)
      
      # get the number of feature of each dataset
      features = X_plus_one.shape[1]

      # training model
      autoencoder, history = training_model(model_architecture, X_plus_one, X_test_one, latent_list[n], verbose = 0, loss = "mse", epochs=epochs, batch_size=64)
      
      # getting predictions of the training data
      preds = autoencoder.predict(X_plus_one)
      # calculate the difference between preds and the training data using (input - preds)**2
      pred_errors = tf.math.square(tf.math.subtract(preds, X_plus_one))
      # append pred_errors to the avg_pred_errors_list
      pred_errors_list.append(pred_errors)

      # getting predictions (recontructions) of the test data
      preds_test = autoencoder.predict(X_test_one)
      # calculate the difference between preds and the test data using (input - preds)**2. These values is used to create a new feature added to the next training loop
      pred_errors_test = tf.math.square(tf.math.subtract(preds_test, X_test_one))
      # append avg_preds_test to the test_errors_list
      test_errors_list.append(pred_errors_test)

      # calculate auc score
      errors_test = tf.keras.losses.mse(preds_test, X_test_one)
      errors_list.append(errors_test)
      # calculate the average of all elements in test_errors_list up to index i. This average value is used to calculate AUC Scores
      tensor_errors_list  = tf.reshape(tf.math.reduce_mean(tf.convert_to_tensor(errors_list), axis=0), [-1, 1])

      # 4. Calculating AUC score of each training model 
      # calculate the average of pred_errors_test. We set axis = 1, so that the average is calculated by column-weise
      auc_score = roc_auc_score(y_test, tensor_errors_list)
      auc_list.append(auc_score)

    fullsyn_dict['auc'].append(auc_list)

  # create a dictionary contains all AUC scores of all datasets corresponding to the number of autoencoders in the ensemble
  fullsyn_auc = {'number of Autoencoders': list(range(1, num_models + 1)),
                 'gas-drift': fullsyn_dict['auc'][0]}  
  # save result to .csv and export it
  fullsyn_df = pd.DataFrame.from_dict(fullsyn_auc)
  fullsyn_df.to_csv(DATA_PATH_OUTPUT / 'synchronizing' / 'gas_10models_synchronizing.csv')
  return fullsyn_df

#LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112]
test_synchroniying = test_synchronizing(Autoencoder, num_models=7, data=[gas_drift], legend=['gas-drift'], latent_list=[112], epochs=2000)
