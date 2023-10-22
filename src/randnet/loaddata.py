import numpy as np
from pathlib import *

# Paths for data input and output directories
DATA_PATH_BASE: Path = Path(__file__).parent.parent.parent / 'data'
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
def loaddata(dataset):
    x = dataset['x']
    tx = dataset['tx']
    ty= dataset['ty']
    return x,tx,ty


#if __name__=="__main__":
#    x,tx,ty=loaddata(dataset=cardio)
#    print(x.shape,tx.shape,ty.shape)
#

#auc = np.load('auc.npz')
#print(auc['auc'])
