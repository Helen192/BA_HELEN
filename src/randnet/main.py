from randnet import RandNet
import pandas as pd

from loaddata import *

auc_dict = {'dataset': LEGEND, 'AUC':[]}

for df in DATA:
    x,tx,ty=loaddata(df)
    r=RandNet(lr=0.01,normalise=False)
    auc=r.train(x,tx,ty)
    auc_dict['AUC'].append(auc)

auc_df = pd.DataFrame.from_dict(auc_dict)
auc_df.to_csv('Randnet_auc.csv')

#r=RandNet(lr=0.01,normalise=False)
#auc=r.train(x,tx,ty)
#print(auc)
#with open("auc.json","w") as f:
#    f.write(str(auc))
#np.savez_compressed("auc.npz",auc=auc)