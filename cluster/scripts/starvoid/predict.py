from csbdeep.models import Config, CARE
import numpy as np

from scipy import ndimage

from os.path import join
from skimage import io

import json

def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    return x*std + mean

with open('experiment.json', 'r') as f:
    exp_params = json.load(f)

files = np.load(exp_params["test_path"])
X_test = files['X_test']


train_files = np.load(exp_params["train_path"])
X_trn = train_files['X_train']

mean, std = np.mean(X_trn), np.std(X_trn)
X = normalize(X_test, mean, std)

model = CARE(None, name= exp_params['model_name'], basedir= exp_params['base_dir'])

for i in range(X.shape[0]):
    predicton = model.predict(X[i][..., np.newaxis], axes='YXC',normalizer=None )
    prediction_exp = np.exp(predicton[...,1:])
    prediction_seg = prediction_exp/np.sum(prediction_exp, axis = 2)[...,np.newaxis]
    predicton_denoise = denormalize(predicton[...,0], mean, std)
    prediction_seg = prediction_seg[...,2]
    pred_thresholded = predicton[...,2]>0.5
    labels, nb = ndimage.label(pred_thresholded)
#    predictions.append(pred)
    io.imsave(join(exp_params['base_dir'], 'mask'+str(i).zfill(3)+'.tif'), labels)
    io.imsave(join(exp_params['base_dir'], 'foreground'+str(i).zfill(3)+'.tif'), prediction_seg)
