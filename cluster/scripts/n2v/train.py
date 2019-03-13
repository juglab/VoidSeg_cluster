#Imports

from n2v.models import Config, CARE
import numpy as np
from n2v.utils import plot_some, plot_history
from n2v.utils.n2v_utils import manipulate_val_data

from matplotlib import pyplot as plt

import urllib
import pickle
import os
import zipfile
import json
from os.path import join

def cutHalf(data, repeat):
    data_=data
    print(data_.shape)
    for i in range(repeat):
        newsize=int(data_.shape[1]/2)
        a=data_[:,:newsize,:]
        b=data_[:,newsize:,:]
  #      print(a.shape,b.shape)
        data_=np.concatenate((a,b), axis=0)

        newsize=int(data_.shape[2]/2)
        a=data_[:,:,:newsize]
        b=data_[:,:,newsize:]
        data_=np.concatenate((a,b), axis=0)
        print(data_.shape)
    return data_

def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    return x*std + mean


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)


#Read training images and GT from StarVoid/dataset/...
train_files = np.load(exp_params["train_path"])
X_trn = train_files['X_train']

train_frac = int(np.round((exp_params['train_frac']/100)*X_trn.shape[0]))

X = X_trn[:train_frac]
print('Training data size', X.shape)
if 'augment' in exp_params.keys():
    if  exp_params['augment']:
        print('augmenting training data')
        X_=X.copy()
        X=np.concatenate((X, np.rot90(X_,2,(1,2))) )
        X=np.concatenate((X, np.flip(X)) )
        print('Training data size after augmentation', X.shape)

X_val = train_files['X_val']
if 'CTC' in  exp_params['exp_name']:
    X_val = cutHalf(X_val[:640,:640],2)

#Config from json file
with open(exp_params['model_name']+'/config.json', 'r') as f:
    conf = json.load(f)

model = CARE(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])
print(conf)

num_pix = conf['n2v_num_pix']
pixelsInPatch = conf['n2v_patch_shape'][0]*conf['n2v_patch_shape'][1]


X = X[...,np.newaxis]

mean, std = np.mean(X), np.std(X)
X = normalize(X, mean, std)

# We concatenate an extra channel filled with zeros. It will be internally used for the masking.
Y = np.concatenate((X, np.zeros(X.shape, dtype=np.float32)), axis=3)

X_val = X_val[...,np.newaxis]
X_val = normalize(X_val, mean, std)

# 1. Option
Y_val = np.concatenate((X_val.copy(), np.zeros(X_val.shape, dtype=np.float32)), axis=3)
manipulate_val_data(X_val, Y_val,num_pix=int(num_pix*X_val.shape[1]*X_val.shape[2]/float(pixelsInPatch)) , shape=(X_val.shape[1], X_val.shape[2]))

# 2. Option
#Y_val = np.concatenate((X_val.copy(), np.ones(X_val.shape)), axis=3)

# The validation set is noisy as well:


#%%capture train_log
hist = model.train(X,Y,validation_data=(X_val,Y_val))

with open(join(exp_params['base_dir'], exp_params['model_name'], 'history_'+exp_params['model_name']+'.dat'), 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
