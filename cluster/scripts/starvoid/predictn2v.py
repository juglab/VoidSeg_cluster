from csbdeep.models import Config, CARE
import numpy as np

from scipy import ndimage

from os.path import join
from skimage import io

import pickle

import json

class PredictN2V:
    
    def __init__(self, exp_params, model):
        self.exp_params = exp_params
        self.model = model

    def normalize(self, img, mean, std):
        zero_mean = img - mean
        return zero_mean/std

    def denormalize(self, x, mean, std):
        return x*std + mean

    def predict(self,X_train, X_val, X_test, model_id, mean, std):
        predictions = []
        
        # model = CARE(None, name= self.exp_params['model_name']+str(model_id), basedir= self.exp_params['base_dir'])
        for i in range(X_train.shape[0]):
            predictions.append(self.denormalize(self.model.predict(X_train[i], axes='YX',normalizer=None ), mean, std))
        X_train_d = np.array(predictions)
        
        predictions = []
        # Denoise all images
        for i in range(X_val.shape[0]):
            predictions.append(self.denormalize(self.model.predict(X_val[i], axes='YX',normalizer=None ), mean, std))
        X_val_d = np.array(predictions)
        
        predictions = []
        # Denoise all images
        for i in range(X_test.shape[0]):
            predictions.append(self.denormalize(self.model.predict(X_test[i], axes='YX',normalizer=None ), mean, std))
        X_test_d = np.array(predictions)
        
        return X_train_d, X_val_d, X_test_d
                
    


        
    

    

    
