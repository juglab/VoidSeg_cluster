#Imports
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import pickle

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes
from stardist import Config, StarDist, StarDistData
import json
from os.path import join

with open('experiment.json', 'r') as f:
    exp_params = json.load(f)


#Read training images and GT from StarVoid/dataset/...
files = np.load(exp_params["train_path"])

X_trn = files['X_train']
Y_trn = files['Y_train']

train_frac = int(np.round(exp_params['train_frac']/100.0*X_trn.shape[0]))

X_trn = X_trn[:train_frac]
Y_trn = Y_trn[:train_frac]

print('Training data size', X_trn.shape)
if 'augment' in exp_params.keys():
    if  exp_params['augment']:
        print('augmenting training data')
        X_trn=np.concatenate((X_trn, np.rot90(X_trn,2,(1,2))) )
        X_trn=np.concatenate((X_trn, np.flip(X_trn)) )
        Y_trn=np.concatenate((Y_trn, np.rot90(Y_trn,2,(1,2))) )
        Y_trn=np.concatenate((Y_trn, np.flip(Y_trn)) )
        print('Training data size after augmentation', X_trn.shape)

X_val = files['X_val']
Y_val = files['Y_val']
#Image normalization and hole filling in labels
X_trn = [normalize(x,1,99.8) for x in X_trn]
Y_trn = [fill_label_holes(y) for y in Y_trn]
X_val = [normalize(x,1,99.8) for x in X_val]
Y_val = [fill_label_holes(y) for y in Y_val]

model = StarDist(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])

#%%capture train_log
hist = model.train(X_trn,Y_trn,validation_data=(X_val,Y_val))

with open(join(exp_params['base_dir'], exp_params['model_name'], 'history_'+exp_params['model_name']+'.dat'), 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
