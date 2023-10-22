# This module contains all the basic functions to get an autoencoder model. This module can be used from other modules, 
# so that we do not have to write again and again reusable functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from pathlib import *

# Getting data

# Paths for data input and output directories
DATA_PATH_BASE: Path = Path(__file__).parent.parent / 'data'
DATA_PATH_INPUT: Path = DATA_PATH_BASE / 'input'
DATA_PATH_OUTPUT: Path = DATA_PATH_BASE / 'output'


cardio = np.load(DATA_PATH_INPUT / 'cardio.npz')
gas_drift = np.load(DATA_PATH_INPUT / 'gas-drift.npz')
satellite = np.load(DATA_PATH_INPUT / 'satellite.npz')
magic_telescope = np.load(DATA_PATH_INPUT / 'MagicTelescope.npz')
pendigits = np.load(DATA_PATH_INPUT / 'pendigits.npz')
phoneme = np.load(DATA_PATH_INPUT / 'phoneme.npz')
pollen = np.load(DATA_PATH_INPUT / 'pollen.npz')
speech = np.load(DATA_PATH_INPUT / 'speech.npz')
waveform = np.load(DATA_PATH_INPUT / 'waveform-5000.npz')

# list of data, legend and the best suitable latent space for each data
DATA = [cardio, gas_drift, satellite, magic_telescope, pendigits, phoneme, pollen, speech, waveform]
LEGEND = ['cardio', 'gas_drift', 'satellite', 'magic_telescope', 'pendigits', 'phoneme', 'pollen', 'speech', 'waveform']
#LATENT_LIST = [4, 112, 64, 8, 8, 8, 8, 8, 8]



def info(data:list, legend: list):
  """
  This function prints out an overview of all training datasets in data list in a dataframe
  
  Arguments:
    data -- a list of datasets
  
  Returns:
    a dataframe with name of dataset, shape, ndim, min, max
  """

  df = {'dataset':LEGEND,
        'shape': [],
        'ndim': [],
        'min': [],
        'max':[]}
  for da in data:
    df['shape'].append(da['x'].shape)
    df['ndim'].append(da['x'].ndim)
    df['min'].append(da['x'].min())
    df['max'].append(da['x'].max())
  info = pd.DataFrame.from_dict(df)
  return info

print(info(DATA, LEGEND))

def scaler(X_tra, X_te, scaler="minmax"):
  """
  Feature scaling using MinMaxScaler oder StandardScaler

  Arguments:
  X_tra -- traing dataset before scaling
  X_te -- test dataset before scaling
  scaler -- scaler methods: minmax = MinMaxScaler; standard = StandardScaler

  Returns:
  X_train -- scaled training set
  X_test -- scaled test set
  """
  from sklearn.preprocessing import MinMaxScaler, StandardScaler

  if scaler == "standard":
    sc = StandardScaler()
  else:
    sc = MinMaxScaler(feature_range=(0, 1))
  
  X_train = sc.fit_transform(X_tra.copy())
  X_test = sc.transform(X_te.copy())
  return X_train, X_test

# Create a model by subclassing Model class in tensorflow
class Autoencoder(Model):
  """
  An autoencoder with Encoder and decoder blocks and adjustable size of laten space
  
  Arguments:
    input_dim -- number of NN units at layer 0 (input)
    latent -- size of laten space layer
         
  Returns: 
    autoencoder -- autoencoder Model
  """

  def __init__(self, input_dim, latent):
    super().__init__()
    # encoder block
    self.encoder = Sequential([
        Dense(latent * 8, activation='relu'),
        Dropout(0.1),
        Dense(latent * 4, activation='relu'),
        Dropout(0.1),
        Dense(latent * 2, activation='relu'),
        Dropout(0.1),
        Dense(latent, activation='relu')
    ])
    # Decoder block
    self.decoder = Sequential([
        Dense(latent * 2, activation='relu'),
        Dropout(0.1),
        Dense(latent * 4, activation='relu'),
        Dropout(0.1),
        Dense(latent * 8, activation='relu'),
        Dropout(0.1),
        Dense(input_dim, activation='sigmoid')
    ])

  def call(self, inputs):
    encode = self.encoder(inputs)
    decode = self.decoder(encode)
    return decode
  

def training_model(model_architecture, X_train, X_test, latent = 4, verbose = 0, loss = "mse", epochs=50, batch_size=64):
  """
  training a model

  Arguments:
    model_architecture -- architecture of model, which is used to train model
    X_train  -- input data
    X-test -- test data
    latent -- the size of latent space applied for model_architecture
    loss -- the loss metrics is used for training model
    epochs -- number of training loops
    batch_size
    verbose -- showing progress of trainin model

  Returns:
    autoencoder -- the trained model
    history -- the history of trained model
  """
  # Set random seed
  #tf.keras.utils.set_random_seed(42)

  # Create autoencoder model
  autoencoder = model_architecture(input_dim=X_train.shape[1], latent = latent)

  # callback will stop the training when there is no improvement in the validation loss for 50 consecutive epochs
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50)   #  Training will stop when the val_loss starts to increase 

  # Loss and optimizer definition
  autoencoder.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=loss,  #  Computes the mean of squares of errors between y_true & y_pred.
                      metrics=[loss]) 

  # Fit the autoencoder
  history= autoencoder.fit(x=X_train, 
                          y=X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = verbose,
                          callbacks = [callback],
                          validation_split=0.2,   # Fraction of the training data to be used as validation data
                          shuffle=True)
  return autoencoder, history

  
def extract_data(dataset):
  """
  This function extract X_train, X_test, y_test from a dataset and apply MinMaxScaler on X_train, X_test

  Arguments:
    dataset -- a dataset contains features x, tx, ty

  Returns:
  X_train: scaled x
  X_test: scaled tx
  y_test: ty
  """
  X_tra = dataset['x']
  X_te = dataset['tx']
  y_test= dataset['ty']
  # Feature scaling
  X_train, X_test = scaler(X_tra, X_te, scaler="minmax")
  return X_train, X_test, y_test

def scaling_factor(X_train, alpha=0.1):
  features = X_train.shape[1]
  d_factor = -(alpha * features) / (alpha - 1)
  return d_factor

def display(data:list, legend: list):
  df = {'dataset':LEGEND,
        'X_train': [],
        'X_test': [],
        'Y_test': [],
        'anomalies_test': []}
  
  for da in data:
    df['X_train'].append(da['x'].shape)
    df['X_test'].append(da['tx'].shape)
    df['Y_test'].append(da['ty'].shape)
    df['anomalies_test'].append(sum(da['ty']))

  info = pd.DataFrame.from_dict(df)
  return info

print(display(DATA, LEGEND))
