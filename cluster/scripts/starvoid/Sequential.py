import Scheme
import numpy as np
from csbdeep.models import CARE
from train_utils import  *
from os.path import join
from os.path import dirname
import pickle


class Sequential(Scheme.Scheme):

    def load_n2v_train_test_data(self):
        n2v_train_data = np.load(self.exp_conf['train_path'])
        n2v_test_data = np.load(self.exp_conf['test_path'])
        return n2v_train_data, n2v_test_data

    def load_n2v_model(self):
        model = CARE(None, name=self.exp_conf['model_name']+'_denoise', basedir=self.exp_conf['base_dir'])
        return model

    def preprocess_n2v(self, train_data, test_data):
        X_train = train_data['X_train']
        Y_train = train_data['Y_train']
        X_test = test_data['X_test']
        Y_test = test_data['Y_test']

        print(X_train.shape, X_test.shape)

        # if(X_test.shape[1]==X_train.shape[1] and X_test.shape[2]==X_train.shape[2]):
        #     X_test_patches = X_test
        #     Y_test_patches = Y_test
        # else:
        for image_num in range(X_test.shape[0]):
            patchesimages, patchesmasks = create_patches(X_test[image_num], Y_test[image_num], X_train.shape[1])
            if(image_num == 0):
                X_test_patches = patchesimages
                Y_test_patches = patchesmasks
            else:
                X_test_patches = np.concatenate((X_test_patches, patchesimages))
                Y_test_patches = np.concatenate((Y_test_patches, patchesmasks))

        X_train_N2V = np.concatenate((X_train, X_test_patches))
        Y_train_N2V = np.concatenate((Y_train, Y_test_patches))
        
        return X_train_N2V, Y_train_N2V

    def load_seg_model(self):
        model = CARE(None, name=self.exp_conf['model_name']+'_seg', basedir=self.exp_conf['base_dir'])
        return model

    def load_seg_train_test_data(self):
        seg_train_data = np.load(self.exp_conf['train_path'])
        seg_test_data = np.load(self.exp_conf['test_path'])
        return seg_train_data, seg_test_data

    def preprocess_seg(self, n2v_model, n2v_train_data, n2v_test_data, mean_std_denoise):
        clean_train, clean_val, clean_test = self.predict_denoise(n2v_model, n2v_train_data, n2v_test_data, mean_std_denoise)
        clean_train = clean_train[:, :, :, 0]
        clean_val = clean_val[:, :, :, 0]
        # if (clean_test.shape[1] == clean_train.shape[1] and clean_test.shape[2] == clean_train.shape[2]):
        #     clean_test = clean_test[:,:,:,0]
        # else:
        for test_img_num in range(clean_test.shape[0]):  # Since test set has images of different shapes, zeroing out segmentation chnnel is done individually for each image
            clean_test[test_img_num] = clean_test[test_img_num][:, :, 0]
        seg_train_data = {}
        seg_train_data['X_train'] = clean_train
        seg_train_data['X_val'] = clean_val
        seg_train_data['Y_train'] = n2v_train_data['Y_train']
        seg_train_data['Y_val'] = n2v_train_data['Y_val']
        seg_test_data = {}
        seg_test_data['X_test'] = clean_test
        np.savez_compressed(dirname(self.exp_conf['train_path']) + '/N2V_TrainVal_'+self.exp_conf['exp_name']+'.npz', X_train=clean_train, Y_train=n2v_train_data['Y_train'], X_val=clean_val, Y_val=n2v_train_data['Y_val'])
        np.savez_compressed(dirname(self.exp_conf['test_path']) + '/N2V_Test_'+self.exp_conf['exp_name']+'.npz', X_test=clean_test)
        return seg_train_data, seg_test_data
        
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


