import Scheme
import numpy as np
from csbdeep.models import CARE
from sklearn.feature_extraction import image
from os.path import join
import pickle


class Sequential(Scheme.Scheme):
    
    def create_patches(images, masks, size):
        patchesimages = image.extract_patches_2d(images, (size, size), 10, 0)  
        patchesmasks = image.extract_patches_2d(masks, (size, size), 10, 0)
    
        return patchesimages, patchesmasks

    def load_n2v_train_test_data(self):
        n2v_train_data = np.load(self.exp_conf['train_path'])
        n2v_test_data = np.load(self.exp_conf['test_path'])
        return n2v_train_data, n2v_test_data

    def load_n2v_model(self):
        model = CARE(None, name=self.exp_conf['model_name']+'_denoise', basedir=self.exp_conf['base_dir'])
        return model

    def preprocess(self, model, train_data, val_data, test_data):  
        X_train = train_data['X_train']
        Y_train = train_data['Y_train']
        X_test = test_data['X_test']
        Y_test = test_data['Y_test']
        
        for image_num in range(test_data.shape[0]):
            patchesimages, patchesmasks = create_patches(X_test[image_num], Y_test[image_num], X_train.shape[1])
            if(image_num == 0):
                X_test_patches = patchesimages
                Y_test_patches = patchesmasks
            else:
                X_test_patches = np.concatenate((X_test_patches, patchesimages))
                Y_test_patches = np.concatenate((Y_test_patches, patchesmasks))

        X_train_N2V = np.concatenate((X_train, X_test_patches))
        Y_train_N2V = np.concatenate((Y_train, Y_test_patches))
        
        return X_train_N2V, Y_trainN2V

    def load_seg_train_data(self):
        data = np.load(self.exp_conf['train_path'])
        return data

    def load_seg_model(self):
        model = CARE(None, name=self.exp_conf['model_name']+'_seg', basedir=self.exp_conf['base_dir'])
        return model

    def load_seg_test_data(self):
        return np.load(self.exp_conf['test_path'])
        
    def train_denoise(self, model, train, val):
        hist = model.train(train[0], train[1], validation_data=val)

        with open(join(self.exp_conf['base_dir'], self.exp_conf['model_name']+str('_denoise'), 'history_' + self.exp_conf['model_name'] + str('_denoise')+'.dat'),
                  'wb') as file_pi:
              pickle.dump(hist.history, file_pi)

    def train_seg(self, model, train, val):
        hist = model.train(train[0], train[1], validation_data=val)

        with open(join(self.exp_conf['base_dir'], self.exp_conf['model_name']+str('_seg'), 'history_' + self.exp_conf['model_name'] + str('_seg')+'.dat'),
                  'wb') as file_pi:
              pickle.dump(hist.history, file_pi)