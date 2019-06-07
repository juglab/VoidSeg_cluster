from csbdeep.models import Config, CARE
import numpy as np
from csbdeep.utils.n2v_utils import manipulate_val_data
from skimage.segmentation import find_boundaries
import json
from os.path import join
import pickle

class TrainSeg:
    
    def __init__(self, seg_conf, exp_params, load_weights = False):
        self.seg_conf = seg_conf
        self.exp_params = exp_params
        self.load_weights = load_weights
        self.num_pix = seg_conf['n2v_num_pix']
        self.pixelsInPatch = seg_conf['n2v_patch_shape'][0] * seg_conf['n2v_patch_shape'][1]


    def add_boundary_label(self, lbl, dtype=np.uint16):
        """ lbl is an integer label image (not binarized) """
        b = find_boundaries(lbl, mode='outer')
        res = (lbl > 0).astype(dtype)
        res[b] = 2
        return res


    def onehot_encoding(self, lbl, n_classes=3, dtype=np.uint32):
        """ n_classes will be determined by max lbl value if its value is None """
        from keras.utils import to_categorical
        onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
        for i in range(n_classes):
            onehot[lbl == i, ..., i] = 1
        return onehot


    def normalize(self, img, mean, std):
        zero_mean = img - mean
        return zero_mean / std


    def denormalize(self, x, mean, std):
        return x * std + mean


    def convert_to_oneHot(self, data):
        data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
        for i in range(data.shape[0]):
            data_oneHot[i] = self.onehot_encoding(self.add_boundary_label(data[i].astype(np.int32)))
        return data_oneHot
        
    def prepare_data_and_segment(self, X_train, Y_train, X_val, Y_val):

        mean, std = np.mean(X_train), np.std(X_train)
        X_train = self.normalize(X_train, mean, std)
        X_val = self.normalize(X_val, mean, std)
        Y_train_oneHot = self.convert_to_oneHot(Y_train)
        Y_val_oneHot = self.convert_to_oneHot(Y_val)
        Y_train = np.concatenate((X_train[..., np.newaxis], np.zeros(X_train.shape, dtype=np.float32)[...,np.newaxis], Y_train_oneHot), axis=3)
        X_train, Y_train = self.shuffle_train_data(X_train, Y_train)
        X_train, Y_train = self.fractionate_train_data(X_train, Y_train)
        X_train_aug, Y_train_aug = self.augment_train_data(X_train, Y_train)
        X_validation, Y_validation = self.prepare_val_data(X_val, Y_val, Y_val_oneHot)
        X_validation_aug, Y_validation_aug = self.augment_val_data(X_validation, Y_validation)
        self.build_model(X_train_aug, Y_train_aug, X_validation_aug, Y_validation_aug)
    
    
    def shuffle_train_data(self, X_train, Y_train):
        if 'is_seeding' in self.exp_params.keys():
            if self.exp_params['is_seeding']:
                print('seeding training data')
                np.random.seed(self.exp_params['random_seed'])
                seed_ind = np.random.permutation(X_train.shape[0])
                X_train = X_train[seed_ind]
                Y_train = Y_train[seed_ind]
                
        return X_train, Y_train
        
    def fractionate_train_data(self, X_train, Y_train):
        train_frac = int(np.round((self.exp_params['train_frac'] / 100) * X_train.shape[0]))
        X_train = X_train[:train_frac]
        Y_train = Y_train[:train_frac]
        
        return X_train, Y_train
            
    def augment_train_data(self, X_train, Y_train):
    
        if 'augment' in self.exp_params.keys():
            if self.exp_params['augment']:
                print('augmenting training data')
                X_ = X_train.copy()
                X_train_aug = np.concatenate((X_train, np.rot90(X_, 2, (1, 2))))
                X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1), np.flip(X_train_aug, axis=2)))
                Y_ = Y_train.copy()
                Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 2, (1, 2))))
                Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1), np.flip(Y_train_aug, axis=2)))
                print('Training data size after augmentation', X_train_aug.shape)
                print('Training data size after augmentation', Y_train_aug.shape)
        
        return X_train_aug, Y_train_aug
            
    def prepare_val_data(self, X_val, Y_val, Y_val_oneHot):
        
        #Prepare data for validation in joint scheme. 
        X_validation = X_val[...,np.newaxis]
        Y_validation = Y_val[...,np.newaxis]
        Y_validation = np.concatenate((X_validation.copy(), np.ones(X_validation.shape, dtype=np.float32)), axis=3) ## Please check here, np.ones???
        # manipulate_val_data(X_validation, Y_validation, num_pix=int(self.num_pix * X_validation.shape[1] * X_validation.shape[2] / float(self.pixelsInPatch)),
 #                    shape=(X_validation.shape[1], X_validation.shape[2]))
        Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)
        
        return X_validation, Y_validation
    

    def augment_val_data(self, X_validation, Y_validation):
    
        # Augment validation
        if 'augment' in self.exp_params.keys():
            if self.exp_params['augment']:
                print('augment validation data')
                X_ = X_validation.copy()
                X_validation_aug = np.concatenate((X_validation, np.rot90(X_, 2, (1, 2))))
                X_validation_aug = np.concatenate(
                    (X_validation_aug, np.flip(X_validation_aug, axis=1), np.flip(X_validation_aug, axis=2)))
                Y_ = Y_validation.copy()
                Y_validation_aug = np.concatenate((Y_validation, np.rot90(Y_, 2, (1, 2))))
                Y_validation_aug = np.concatenate(
                    (Y_validation_aug, np.flip(Y_validation_aug, axis=1), np.flip(Y_validation_aug, axis=2)))
                    
        return X_validation_aug, Y_validation_aug
                
        
    def build_model(self, X_train_aug, Y_train_aug, X_validation_aug, Y_validation_aug):
                
        model = CARE(None, name= self.exp_params['model_name']+str('_seg_model'), basedir= self.exp_params['base_dir'])
        print(self.seg_conf)
        
        if(self.load_weights):
            model.load_weights('../' +self.exp_params['model_name']+str('_n2v_init_model')+'/weights_best.h5')  #It will always be model_n2v_init_model for initialization

        hist = model.train(X_train_aug[..., np.newaxis],Y_train_aug,validation_data=(X_validation_aug,Y_validation_aug))

        with open(join(self.exp_params['base_dir'], self.exp_params['model_name']+str('_seg_model'), 'history_' + self.exp_params['model_name'] + str('_seg_model')+'.dat'),
                  'wb') as file_pi:
              pickle.dump(hist.history, file_pi)
    
