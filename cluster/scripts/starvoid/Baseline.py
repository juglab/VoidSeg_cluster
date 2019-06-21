import Scheme
import numpy as np
from csbdeep.models import CARE
from os.path import join
import pickle


class Baseline(Scheme.Scheme):

    def load_n2v_train_test_data(self):
        # Nothing to do
        return None, None

    def create_n2v_train_data(self, train_data_x, train_data_y, n2v_train_data):
        # Nothing to do
        return None, None, None

    def load_n2v_model(self):
        # Nothing to do
        return None

    def preprocess_n2v(self, train_data, test_data):
        # Noting to do
        return None, None
    
    def train_denoise(self, model, train_data, val_data):
        # Noting to do
        pass
        
    def predict_denoise(self, model, test_data):
        # Noting to do
        pass

    def load_seg_train_test_data(self):
        train_data = np.load(self.exp_conf['train_path'])
        test_data = np.load(self.exp_conf['test_path'])
        return train_data, test_data

    def preprocess_seg(self, n2v_model, seg_train_data, seg_test_data, mean_std_denoise):
        return np.load(self.exp_conf['train_path']), np.load(self.exp_conf['test_path'])

    def load_seg_model(self):
        model = CARE(None, name=self.exp_conf['model_name']+'_seg', basedir=self.exp_conf['base_dir'])
        return model

    def train_seg(self, model, train, val):
        hist = model.train(train[0], train[1], validation_data=val)

        with open(join(self.exp_conf['base_dir'], self.exp_conf['model_name']+str('_seg'), 'history_' + self.exp_conf['model_name'] + str('_seg')+'.dat'),
                  'wb') as file_pi:
              pickle.dump(hist.history, file_pi)


