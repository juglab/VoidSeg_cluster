from csbdeep.models import Config, CARE
import numpy as np

from scipy import ndimage

from os.path import join
from skimage import io

import pickle

import json

class PredictSeg:
    
    def __init__(self, exp_params):
        self.exp_params = exp_params

    def normalize(self, img, mean, std):
        zero_mean = img - mean
        return zero_mean/std

    def denormalize(self, x, mean, std):
        return x*std + mean

    def predict(self,X_test, X_trn, model_id):
        mean, std = np.mean(X_trn), np.std(X_trn)
        X = self.normalize(X_test, mean, std)
        model = CARE(None, name= self.exp_params['model_name']+str(model_id), basedir= self.exp_params['base_dir']) 
        with open('best_score.dat', 'rb') as best_score_file: 
            ts = pickle.load(best_score_file)[0]
        print('Use threshold =', ts)
        for i in range(X.shape[0]):
            prediction = model.predict(X[i], axes='YX',normalizer=None )
            denoised = prediction[...,0]
            prediction_exp = np.exp(prediction[...,1:])
            prediction_seg = prediction_exp/np.sum(prediction_exp, axis = 2)[...,np.newaxis]
            predicton_denoise = self.denormalize(denoised, mean, std)
            prediction_bg = prediction_seg[...,0]
            prediction_fg = prediction_seg[...,1]
            prediction_b = prediction_seg[...,2]
            pred_thresholded = prediction_fg>ts
            labels, nb = ndimage.label(pred_thresholded)
            #    predictions.append(pred)
            io.imsave(join(self.exp_params['base_dir'], 'mask'+str(i).zfill(3)+'.tif'), labels.astype(np.int16))
            io.imsave(join(self.exp_params['base_dir'], 'foreground'+str(i).zfill(3)+'.tif'), prediction_fg)
            io.imsave(join(self.exp_params['base_dir'], 'background'+str(i).zfill(3)+'.tif'), prediction_bg)
            io.imsave(join(self.exp_params['base_dir'], 'border'+str(i).zfill(3)+'.tif'), prediction_b)

                
    


        
    

    

    
