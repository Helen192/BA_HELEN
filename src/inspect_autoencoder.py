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


#LATENT_LIST = [21, 112, 192, 112, 16, 96, 32, 112, 112]



def inspect_autoencoder(model_architecture, latent_list: list, data: list, legend: list, epochs):
  """
  This function calculate auc scores of datasets using autoencoder model and  show us the ROC curve and AUC score 

  Arguments:
    model_architecture -- the model architecture that we want to use to create model
    data -- a list of datasets
    legend -- name of datasets used to display on plot
  Returns:
    aut_dict -- this dictionary contains auc scores of datasets
  """
  # Set random seed
  tf.keras.utils.set_random_seed(42)
  
  aut_dict = {'dataset': legend, 'latent': latent_list, 'auc': []}

  for n in range(len(data)):
    df = data[n] # extract dataset n in data
    # extract X_train, X_test, y_test in df and scaled X_train, X_test
    X_train, X_test, y_test = extract_data(df)

    # create and train autoencoder for each dataset
    autoencoder, history = training_model(model_architecture, X_train, X_test, latent = latent_list[n], batch_size=64,epochs=epochs)
    # getting predictions (recontructions) of the test data
    preds = autoencoder.predict(X_test)
    # calculate the difference between preds and test data using mean square error
    pred_errors = tf.keras.losses.mse(preds, X_test)
    # Check the prediction performance
    auc_score = roc_auc_score(y_test, pred_errors)
    aut_dict['auc'].append(auc_score)

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, pred_errors)
    plt.plot(fpr, tpr, label = f"{legend[n]}: {auc_score}")
    
  #plt.legend(loc='best')
  #plt.title('Receiver Operating Characteristic using Autoencoder ')
  #plt.plot([0, 1], [0, 1],'r--')
  #plt.ylabel('True Positive Rate')
  #plt.xlabel('False Positive Rate')
  ## save plot
  #plt.savefig(DATA_PATH_OUTPUT / '300epochs_100batch_not_finetun_version2_inspect_autoencoder.jpg')
  #plt.show()
  # save result to .csv and export it
  aut_df = pd.DataFrame.from_dict(aut_dict)
  aut_df.to_csv(DATA_PATH_OUTPUT / 'ver3_unoptimal_autoencoder_500e_64b_auc.csv')

  return aut_df

#plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


#LATENT_LIST = [4, 32, 16, 4, 4, 4, 4, 64, 32]

#LATENT_LIST_Version2 = [4, 16, 16, 4, 4, 4, 4, 256, 32]
unoptimal_latent_ver3 = [32, 64, 32, 8, 8, 4, 4, 128, 64]

#aut_df= inspect_autoencoder(Autoencoder, latent_list = LATENT_LIST, data = DATA, legend = LEGEND, epochs = 2000)
#aut_df= inspect_autoencoder(Autoencoder, latent_list = LATENT_LIST_Version2, data = DATA, legend = LEGEND, epochs = 1000)
#aut_df= inspect_autoencoder(Autoencoder, latent_list = LATENT_LIST_Version2, data = DATA, legend = LEGEND, epochs = 300)
aut_df= inspect_autoencoder(Autoencoder, latent_list = unoptimal_latent_ver3, data = DATA, legend = LEGEND, epochs = 500)
print(aut_df)

# check two datasets phoneme and pollen. These datasets have the same number of features 5. 
# But I try to inspect them by using different laten size. One with a huge latent size of 320 and the pollen is trained with a small latent size = 16
#aut_df_2= inspect_autoencoder(Autoencoder, latent_list = [4, 4], data = [phoneme, pollen], legend = ['phoneme', 'pollen'], epochs = 2000)
#print(aut_df_2)