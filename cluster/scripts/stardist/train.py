#Imports
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import pickle

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

if 'is_seeding' in exp_params.keys():
    if exp_params['is_seeding']:
        print('seeding training data')
        np.random.seed(exp_params['random_seed'])
        seed_ind = np.random.permutation(X_trn.shape[0])
        X_train = X_trn[seed_ind]
        Y_train = Y_trn[seed_ind]

train_frac = int(np.round(exp_params['train_frac']/100.0*X_trn.shape[0]))

X_trn = X_trn[:train_frac]
Y_trn = Y_trn[:train_frac]

print('Training data size', X_trn.shape)
if 'augment' in exp_params.keys():
    if  exp_params['augment']:
        print('augmenting training data')

        X_ = X_trn.copy()

        X_train_aug = np.concatenate((X_trn, np.rot90(X_, 1, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
        X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

        Y_ = Y_trn.copy()

        Y_train_aug = np.concatenate((Y_trn, np.rot90(Y_, 1, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 2, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 3, (1, 2))))
        Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))

        print('Training data size after augmentation', X_train_aug.shape)
        print('Training data size after augmentation', Y_train_aug.shape)

X_val = files['X_val']
Y_val = files['Y_val']

if 'augment' in exp_params.keys():
    if  exp_params['augment']:
        print('augmenting validation data')

        X_ = X_val.copy()

        X_val_aug = np.concatenate((X_val, np.rot90(X_, 1, (1, 2))))
        X_val_aug = np.concatenate((X_val_aug, np.rot90(X_, 2, (1, 2))))
        X_val_aug = np.concatenate((X_val_aug, np.rot90(X_, 3, (1, 2))))
        X_val_aug = np.concatenate((X_val_aug, np.flip(X_val_aug, axis=1)))

        Y_ = Y_val.copy()

        Y_val_aug = np.concatenate((Y_val, np.rot90(Y_, 1, (1, 2))))
        Y_val_aug = np.concatenate((Y_val_aug, np.rot90(Y_, 2, (1, 2))))
        Y_val_aug = np.concatenate((Y_val_aug, np.rot90(Y_, 3, (1, 2))))
        Y_val_aug = np.concatenate((Y_val_aug, np.flip(Y_val_aug, axis=1)))

        print('Training data size after augmentation', X_val_aug.shape)
        print('Training data size after augmentation', Y_val_aug.shape)
#Image normalization and hole filling in labels
X_train_aug = [normalize(x,1,99.8) for x in X_train_aug]
Y_train_aug = [fill_label_holes(y.astype(np.uint16)) for y in Y_train_aug]
#Y_train_aug = [fill_label_holes(y) for y in Y_train_aug]
X_val_aug = [normalize(x,1,99.8) for x in X_val_aug]
Y_val_aug = [fill_label_holes(y.astype(np.uint16)) for y in Y_val_aug]
#Y_val_aug = [fill_label_holes(y) for y in Y_val_aug]

model = StarDist(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])

#%%capture train_log
hist = model.train(X_train_aug,Y_train_aug,validation_data=(X_val_aug,Y_val_aug))

with open(join(exp_params['base_dir'], exp_params['model_name'], 'history_'+exp_params['model_name']+'.dat'), 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


