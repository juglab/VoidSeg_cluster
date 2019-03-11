#Imports

from n2v.models import Config, CARE
import numpy as np
from n2v.utils import plot_some, plot_history
from n2v.utils.n2v_utils import manipulate_val_data
import urllib

import os
import zipfile
import json
from os.path import join
from skimage import io

def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    return x*std + mean


with open('experiment.json', 'r') as f:
    exp_params = json.load(f)


#Read training images and GT from StarVoid/dataset/...
test_files = np.load(exp_params["test_path"])
X = test_files['X_test']

train_files = np.load(exp_params["train_path"])
X_trn = train_files['X_train']

mean, std = np.mean(X_trn), np.std(X_trn)
X = normalize(X, mean, std)

model = CARE(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])

# X = X[...,np.newaxis]

#predictions = []
# Denoise all images
for i in range(X.shape[0]):
    pred = denormalize(model.predict(X[i][..., np.newaxis], axes='YXC',normalizer=None ), mean, std)
#    predictions.append(pred)
    io.imsave(join(exp_params['base_dir'], 'mask'+str(i).zfill(3)+'.tif'), pred)
#predictions = np.array(predictions)
