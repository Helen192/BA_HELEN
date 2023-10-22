# Correlation between AUC score and the number of Autoencoder in an ensemble model

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


# for this function, seed_list is generated randomly

def ensemble_model(model_architecture, seed_list, X_train, X_test, y_test, latent, epochs, plot=True):
  """
  Create an ensemble model, which use an unique architecture, but with different numbers of random seed. 
  By setting a different random seed, we get a new model with the same architecture

  Arguments:
    model_architecture -- architecture of model, which is used to train model
    seed_list -- a list of numbers which are used to set random seed for the model
    X_train -- data which are used to train the model
    X_test -- data which are used to test the trained model
    y_test -- true labels of X_test
    plot -- True: plot the ROC curve. False: does not plot the ROC curve
         
  Returns: 
    avg_preds -- the average of predictions errors (the difference between inputs and their predictions) of all models in ensemble model
    auc_score -- the AUC score of ensemble model
    ROC curve
  """
  # list of pred_errors. Each element contains all the difference between inputs and their predictions of one model
  pred_errors_list = []

  for SEED in seed_list:
    # Set random seed
    tf.keras.utils.set_random_seed(SEED)
    # training model
    autoencoder, history = training_model(model_architecture, X_train, X_test, latent = latent, epochs=epochs)
    
    # getting predictions (recontructions) of the test data
    preds = autoencoder.predict(X_test)
    # calculate the difference between preds and test data using mean square error
    pred_errors = tf.keras.losses.mse(preds, X_test)
    # append pred_error to the pred_errors_list
    pred_errors_list.append(pred_errors)
  
  # convert pred_errors_list to a tensor
  preds_tensor = tf.convert_to_tensor(pred_errors_list)
  # calculate the average of all pred_errors in the pred_errors_list. We set axis=0, so that the average is calculated by row-weise. 
  avg_preds = tf.math.reduce_mean(preds_tensor, axis=0)
  # AUC score of the average pred_errors of all models vs. y_test
  auc_score = roc_auc_score(y_test, avg_preds)
  if not plot:
    print(f"AUC score: {auc_score}")
  else:
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, avg_preds)
    plt.title('ROC of ensemble')
    plt.plot(fpr, tpr, 'g', label = 'AUC = %0.2f' % auc_score)
    plt.legend(loc = 'best')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
  
  return auc_score


def plot_auc_models(model_architecture, num_models_list, latent_list: list, data:list, legend:list, epochs):
  """
  create a plot, showing how this final auc score of ensemble model depends on the number of models used.

  Arguments:
    model_architecture -- architecture of model, which is used to train model
    num_models_list -- a list contains the number of models which are integrated in one ensemble model
    latent_list -- this contains the best size of latent space for each dataset in data respectively
    data -- a list of datasets
    legend -- name of datasets used to display on plot
    epochs -- 
         
  Returns: 
    ROC curve
    num_auto -- a dictionary contains AUC score of each dataset according to each element in num_models_list
  """
  num_auto = {'dataset': legend, 'num_auto': num_models_list, 'auc':[]}
  for n in range(len(data)):
    df = data[n] # extract dataset n in data
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)
    # list of auc score corresponding with the number of models integrated in an ensemble model
    AUC_LIST = []

    for i in num_models_list:
      # Create a random list of seed
      seed_list = np.random.randint(1000, size=i).tolist()
      # Get the auc_score of the ensemble model which is trained by using seed_list
      auc_score = ensemble_model(model_architecture, seed_list, X_train, X_test, y_test, latent_list[n], epochs, plot=False)
      AUC_LIST.append(auc_score)
    num_auto['auc'].append(AUC_LIST)
    # the plot display the correlation between auc score and the number of models integrated in an ensemble model
    plt.plot(num_models_list, AUC_LIST, label=legend[n])
  plt.title('Correlation between AUC score and the number of autoencoders in an ensemble model')
  plt.xlabel('the number of autoencoders in an ensemble model')
  plt.ylabel('AUC score of ensembel models')
  plt.legend(loc='best')
  # save the plot
  #plt.savefig(DATA_PATH_OUTPUT / 'ensemble_number_of_auto.jpg')
  plt.savefig(DATA_PATH_OUTPUT / 'to10models_ensemble_number_of_auto.jpg')
  plt.show()

  # create a dictionary contains all AUC scores of all datasets corresponding to the number of autoencoders in the ensemble
  num_auto_auc = {'number of Autoencoders': num_models_list,
                'cardio': num_auto['auc'][0],
                'gas-drift': num_auto['auc'][1],
                'satellite': num_auto['auc'][2],
                'magic_telescope': num_auto['auc'][3],
                'pendigits': num_auto['auc'][4],
                'phoneme': num_auto['auc'][5],
                'pollen': num_auto['auc'][6],
                'speech': num_auto['auc'][7],
                'waveform': num_auto['auc'][8]}

  # save result to .csv and export it
  num_auto_df = pd.DataFrame.from_dict(num_auto_auc)
  num_auto_df.to_csv(DATA_PATH_OUTPUT / 'upto10_models_ensemble_number_of_auto.csv')
  return num_auto




# Test with more models
num_auto_2 = plot_auc_models(Autoencoder,num_models_list=[i for i in range(1, 11)], latent_list = LATENT_LIST, data = DATA, legend = LEGEND, epochs = 2000)
print(num_auto_2)